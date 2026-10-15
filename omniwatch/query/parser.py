"""Natural Language Query Parser.

Responsibility:
- Parses unstructured NL user queries into structured ParsedQuery schema.
- Extracts intent (exists|find|when|count|path), object phrase, class hint, attributes, location phrases, time expressions.
- Uses LLM (Ollama JSON mode) with deterministic rule-based fallback on parsing/validation failure.
- Complies strictly with docs/01_ARCHITECTURE.md section 4.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from omniwatch.config import OmniWatchConfig, load_config

logger = logging.getLogger(__name__)

# Track active parser fallback state for /api/health
FALLBACK_ACTIVE: bool = False
LAST_FALLBACK_REASON: str | None = None
LAST_PARSER_USED: str = "rules"


def get_last_parser_used() -> str:
    """Returns the name of the parser used in the most recent query."""
    return LAST_PARSER_USED


def get_parser_health_status() -> dict[str, Any]:
    """Returns parser health metadata for /api/health."""
    return {
        "active_parser": LAST_PARSER_USED,
        "fallback_active": FALLBACK_ACTIVE,
        "fallback_reason": LAST_FALLBACK_REASON,
    }


IntentType = Literal["exists", "find", "when", "count", "path"]
TimeKind = Literal["relative", "absolute", "none"]


class QueryTimeConstraint(BaseModel):
    kind: TimeKind = "none"
    start: float | None = None  # epoch seconds UTC (resolved by resolver.py)
    end: float | None = None  # epoch seconds UTC
    raw: str | None = None


class QueryAttributes(BaseModel):
    color: list[str] = Field(default_factory=list)
    carries: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    intent: IntentType = "find"
    object_phrase: str
    object_class_hint: str | None = None
    attributes: QueryAttributes = Field(default_factory=QueryAttributes)
    location_phrases: list[str] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)
    time: QueryTimeConstraint = Field(default_factory=QueryTimeConstraint)
    negation: bool = False
    raw: str


KNOWN_COLORS = {
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
    "orange", "purple", "pink", "brown", "silver", "gold", "beige", "navy",
    "maroon", "violet", "teal"
}

# Clothing / body-part nouns that should NEVER be treated as location phrases.
# Keeps "in a red shirt" from becoming location = "a red shirt".
CLOTHING_NOUNS = {
    "shirt", "t-shirt", "tshirt", "jacket", "coat", "hoodie", "sweater",
    "dress", "skirt", "pants", "jeans", "shorts", "suit", "uniform",
    "hat", "cap", "helmet", "scarf", "gloves", "shoes", "boots", "vest",
    "blouse", "top", "sweatshirt", "raincoat", "trousers", "trench",
    "sneakers", "sandals", "tie", "blazer", "parka",
}

CLASS_HINTS: dict[str, list[str]] = {
    "person": [
        "person", "people", "man", "woman", "men", "women", "pedestrian", "pedestrians",
        "guy", "girl", "child", "children", "individual", "someone", "anyone", "human",
        "adult", "courier", "cyclist", "worker", "intruder", "reader", "news reader",
        "anchor", "news anchor", "reporter", "tv reporter", "presenter", "speaker",
        "driver", "officer", "police", "cop", "guard", "security", "security guard",
        "passerby", "customer", "passenger", "student", "host", "kid", "kids", "baby",
        "bystander", "bicyclist", "walker"
    ],
    "vehicle": [
        "car", "cars", "vehicle", "vehicles", "truck", "trucks", "bus", "buses",
        "van", "vans", "suv", "motorcycle", "motorcycles", "bike", "bikes",
        "bicycle", "bicycles", "sedan", "pickup", "automobile", "courier van",
        "police car", "cab", "taxi", "traffic"
    ],
    "bag": [
        "bag", "bags", "backpack", "backpacks", "handbag", "handbags", "suitcase",
        "suitcases", "box", "boxes", "package", "packages", "luggage", "purse",
        "briefcase", "duffel", "tote", "sack"
    ]
}

CARRIES_KEYWORDS = [
    "backpack", "handbag", "suitcase", "bag", "box", "package",
    "luggage", "purse", "briefcase", "duffel", "tote"
]


class RuleQueryParser:
    """Deterministic regex and keyword-based query parser."""

    @staticmethod
    def parse(raw_query: str) -> ParsedQuery:
        text = raw_query.strip()
        lower = text.lower()

        # 1. Intent extraction
        intent: IntentType = "find"
        if re.search(r"^(how many|count|number of)\b", lower):
            intent = "count"
        elif re.search(r"^(when|what time|at what time)\b", lower):
            intent = "when"
        elif re.search(r"^(where did .* (go|head|move)|path|trajectory|route|track)\b", lower):
            intent = "path"
        elif re.search(r"^(did|was|is|were|has|have|are|any)\b", lower) or re.search(r"\b(is there|was there|did any|has any)\b", lower):
            intent = "exists"
        elif re.search(r"^(show|find|search|locate|get|where is|where are|give me|look for)\b", lower):
            intent = "find"

        # 2. Negation extraction
        negation = bool(re.search(r"\b(not|didn't|did not|wasn't|was not|never|no|without)\b", lower))

        # 3. Camera ID extraction
        camera_ids: list[str] = []
        for match in re.finditer(r"\b(?:cam|camera)\s*(\d+)\b", lower):
            cam_norm = f"cam{match.group(1)}"
            if cam_norm not in camera_ids:
                camera_ids.append(cam_norm)

        # Named cameras in deployment feeds
        if re.search(r"\b(yokohama(?:_cctv)?|yokohama\s+cctv)\b", lower):
            if "yokohama_cctv" not in camera_ids:
                camera_ids.append("yokohama_cctv")

        if re.search(r"\b(public(?:_surveillance)?|public\s+surveillance|public\s+cctv|surveillance\s+tours?)\b", lower):
            if "public_surveillance" not in camera_ids:
                camera_ids.append("public_surveillance")

        # 4. Time extraction
        time_kind: TimeKind = "none"
        time_raw: str | None = None

        rel_time_match = re.search(
            r"\b((?:in the\s+)?(?:last|past)\s+(?:\d+|one|two|three|four|five|ten|thirty|twenty|an|a)?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?))\b"
            r"|\b((?:this\s+(?:morning|afternoon|evening)|today|yesterday|tomorrow)(?:\s+(?:at\s+)?(?:noon|midnight|\d{1,2}(?::\d{2})?(?:\s*[ap]m)?))?)\b"
            r"|\b(at\s+(?:noon|midnight))\b"
            r"|\b(\d+\s*(?:hours?|hrs?|minutes?|mins?)\s+ago)\b",
            lower
        )
        if rel_time_match:
            time_kind = "relative"
            time_raw = rel_time_match.group(0).strip()
        else:
            abs_time_match = re.search(
                r"\b(between\s+\d{1,2}(?::\d{2})?(?::\d{2})?(?:\s*[ap]m)?\s+(?:and|to)\s+\d{1,2}(?::\d{2})?(?::\d{2})?(?:\s*[ap]m)?)\b"
                r"|\b(from\s+\d{1,2}(?::\d{2})?(?::\d{2})?(?:\s*[ap]m)?\s+to\s+\d{1,2}(?::\d{2})?(?::\d{2})?(?:\s*[ap]m)?)\b"
                r"|\b(at\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?)\b"
                r"|\b(around\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?)\b",
                lower
            )
            if abs_time_match:
                time_kind = "absolute"
                time_raw = abs_time_match.group(0).strip()

        # 5. Location extraction
        # Look for prepositional phrases after at/in/near/through/by/into/enter/inside/around
        TEMPORAL_NOUNS = {
            "noon", "midnight", "night", "day", "morning", "afternoon", "evening",
            "dawn", "dusk", "today", "yesterday", "tomorrow", "now", "week", "month", "year"
        }
        location_phrases: list[str] = []
        clean_for_loc = re.sub(r"[?!.,]", "", lower).strip()
        loc_matches = re.finditer(
            r"\b(?:at|in|near|through|by|into|enter|entered|inside|around)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?=\s+(?:and\s+(?:near|at|in|by|into|inside|around|the)|or\s+(?:near|at|in|by|into|inside|around|the)|in the|between|from \d|at \d|last|past|yesterday|this morning|with|carrying|wearing|$)|$)",
            clean_for_loc,
        )
        for m in loc_matches:
            raw_loc = m.group(1).strip()
            # Split conjoined locations like "zone alpha and zone beta"
            candidate_locs = re.split(r"\s+\b(?:and|or)\b\s+", raw_loc)
            for loc in candidate_locs:
                loc = loc.strip()
                # Strip leading article (a/an/the)
                loc = re.sub(r"^(a|an|the)\s+", "", loc).strip()
                # Avoid picking camera IDs or pure time words as locations
                if re.match(r"^(cam\s*\d+|camera\s*\d+)$", loc):
                    continue
                if re.match(r"^(\d{1,2}:\d{2}|\d+\s*minutes?|\d+\s*hours?)$", loc):
                    continue
                loc_words = set(loc.split())
                # Skip temporal words (e.g. "at noon", "in the evening")
                if loc in TEMPORAL_NOUNS or (loc_words & TEMPORAL_NOUNS):
                    continue
                # Skip clothing/apparel phrases — these are attribute descriptors, not places
                if loc_words & CLOTHING_NOUNS:
                    continue
                # Skip purely color+noun combos that match known attribute vocabulary
                non_color_words = [w for w in loc.split() if w not in KNOWN_COLORS]
                if not non_color_words or (len(non_color_words) == 1 and non_color_words[0] in CLOTHING_NOUNS):
                    continue
                if loc and loc not in location_phrases and len(loc) > 1:
                    # Exclude purely functional stop words
                    if loc not in {"front", "back", "middle"}:
                        location_phrases.append(loc)

        # 6. Attributes extraction (colors, carries)
        colors_found: list[str] = []
        for word in re.findall(r"\b[a-z]+\b", lower):
            if word in KNOWN_COLORS and word not in colors_found:
                colors_found.append(word)

        carries_found: list[str] = []
        carries_phrase_match = re.search(r"\b(?:carrying|with|holding|hauling)\s+(?:a\s+|an\s+)?([a-z\s]+?)(?=\s+(?:at|in|near|between|from|during|$))", lower)
        if carries_phrase_match:
            candidate_item = carries_phrase_match.group(1).strip()
            for kw in CARRIES_KEYWORDS:
                if kw in candidate_item and kw not in carries_found:
                    carries_found.append(kw)
        for kw in CARRIES_KEYWORDS:
            if re.search(rf"\b{kw}s?\b", lower) and kw not in carries_found:
                carries_found.append(kw)

        # 7. Class hint
        class_hint: str | None = None
        for hint, keywords in CLASS_HINTS.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", lower):
                    class_hint = hint
                    break
            if class_hint:
                break

        # 8. Object phrase extraction
        # Remove question prefixes, time phrases, location phrases
        clean = text
        clean = re.sub(r"^(did|was|is|were|has|have|show me|find|where did|how many|when did|what time did|search for|locate)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^(there|any|a|an|the)\s+", "", clean, flags=re.IGNORECASE)
        if time_raw:
            clean = re.sub(re.escape(time_raw), "", clean, flags=re.IGNORECASE)
        for loc in location_phrases:
            clean = re.sub(rf"\b(?:at|in|near|through|by|into|enter|entered|inside|around)\s+(?:the\s+)?{re.escape(loc)}\b", "", clean, flags=re.IGNORECASE)
        for cam in camera_ids + ["yokohama_cctv", "yokohama cctv", "yokohama", "public_surveillance", "public surveillance", "public cctv", "surveillance tours"]:
            clean = re.sub(rf"\b(?:at|on|in|from)?\s*{re.escape(cam)}\b", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(pass through|pass by|enter|entered|exit|exited|arrive|arrived|depart|departed|go|gone|seen|observed|detected|appear|appeared|show up|turn up|walk|walking|walked|stand|standing|stop|stopped)\b", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"[?!.,]", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()

        # If indefinite human pronouns were used ("anyone", "someone"), normalize to "person"
        if clean.lower() in {"anyone", "someone", "anybody", "somebody", "any one", "some one"}:
            clean = "person"
            if not class_hint:
                class_hint = "person"

        # If clean object phrase is too stripped, build a coherent object phrase
        if not clean or len(clean) < 2:
            parts = []
            if colors_found:
                parts.append(" ".join(colors_found))
            if class_hint:
                parts.append(class_hint)
            else:
                parts.append("object")
            if carries_found:
                parts.append(f"with {' and '.join(carries_found)}")
            clean = " ".join(parts)


        return ParsedQuery(
            intent=intent,
            object_phrase=clean,
            object_class_hint=class_hint,
            attributes=QueryAttributes(color=colors_found, carries=carries_found, other=[]),
            location_phrases=location_phrases,
            camera_ids=camera_ids,
            time=QueryTimeConstraint(kind=time_kind, raw=time_raw),
            negation=negation,
            raw=raw_query,
        )

    @staticmethod
    def is_confident(parsed: ParsedQuery, raw_query: str) -> bool:
        """Determines if the rule-based parse is confident enough to use without LLM.

        Returns True for deterministic queries with recognized object classes,
        colors, carries attributes, or specific cameras/locations.
        Returns False for ambiguous conversational queries or unrecognized targets.
        """
        # If no recognized class hint and no attributes and object phrase is generic
        if parsed.object_class_hint is None and not parsed.attributes.color and not parsed.attributes.carries:
            if parsed.object_phrase.strip().lower() in {"object", "something", "anything", ""}:
                return False

        # If query has vague conversational expressions like 'suspicious', 'unusual', 'loitering'
        lower = raw_query.lower()
        if re.search(r"\b(suspicious|unusual|anomaly|anomalous|weird|strange|loitering)\b", lower):
            return False

        # If object phrase is empty or single character
        if len(parsed.object_phrase.strip()) < 2:
            return False

        return True


class LLMQueryParser:
    """Ollama-based LLM query parser with JSON schema output."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:1b",
        timeout_s: float = 6.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        candidates = [
            Path(__file__).parent / "prompts" / "parser.txt",
            Path("query/prompts/parser.txt"),
            Path("omniwatch/query/prompts/parser.txt"),
        ]
        for p in candidates:
            if p.exists():
                return p.read_text(encoding="utf-8")
        return (
            "You are an expert CCTV video query parser. Output JSON matching ParsedQuery schema only. "
            "Fields: intent (exists|find|when|count|path), object_phrase, object_class_hint, attributes {color, carries, other}, "
            "location_phrases, camera_ids, time {kind, start, end, raw}, negation, raw."
        )

    def parse(self, raw_query: str) -> ParsedQuery:
        global FALLBACK_ACTIVE, LAST_FALLBACK_REASON, LAST_PARSER_USED
        prompt = f"{self.prompt_template}\n\nQuery: {raw_query}\nJSON:\n"

        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["exists", "find", "when", "count", "path"]},
                "object_phrase": {"type": "string"},
                "object_class_hint": {"type": ["string", "null"]},
                "attributes": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "array", "items": {"type": "string"}},
                        "carries": {"type": "array", "items": {"type": "string"}},
                        "other": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["color", "carries", "other"],
                },
                "location_phrases": {"type": "array", "items": {"type": "string"}},
                "camera_ids": {"type": "array", "items": {"type": "string"}},
                "time": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["relative", "absolute", "none"]},
                        "start": {"type": ["number", "null"]},
                        "end": {"type": ["number", "null"]},
                        "raw": {"type": ["string", "null"]},
                    },
                    "required": ["kind", "start", "end", "raw"],
                },
                "negation": {"type": "boolean"},
                "raw": {"type": "string"},
            },
            "required": [
                "intent", "object_phrase", "object_class_hint", "attributes",
                "location_phrases", "camera_ids", "time", "negation", "raw"
            ],
        }

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                res = client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "format": schema,
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                )
                res.raise_for_status()
                data = res.json()
                content = data.get("message", {}).get("content", "")
                parsed_json = json.loads(content)
                parsed = ParsedQuery.model_validate(parsed_json)
                # Supplement any missing location phrases, camera IDs, or time constraints from rules
                rule_supplement = RuleQueryParser.parse(raw_query)
                if not parsed.location_phrases and rule_supplement.location_phrases:
                    parsed.location_phrases = rule_supplement.location_phrases
                if not parsed.camera_ids and rule_supplement.camera_ids:
                    parsed.camera_ids = rule_supplement.camera_ids
                if parsed.time.kind == "none" and rule_supplement.time.kind != "none":
                    parsed.time = rule_supplement.time

                FALLBACK_ACTIVE = False
                LAST_FALLBACK_REASON = None
                LAST_PARSER_USED = "ollama_fallback"
                return parsed

        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, ValidationError, Exception) as exc:
            FALLBACK_ACTIVE = True
            LAST_FALLBACK_REASON = f"{type(exc).__name__}: {exc}"
            LAST_PARSER_USED = "rules"
            logger.warning(
                "Ollama query parsing fallback failed (%s). Continuing with RuleQueryParser.",
                LAST_FALLBACK_REASON,
            )
            return RuleQueryParser.parse(raw_query)


def parse_query(
    raw_query: str,
    use_llm: bool = True,
    config: OmniWatchConfig | None = None,
) -> ParsedQuery:
    """Parses natural language query into structured ParsedQuery representation.

    Follows the project specification precedence:
    1. Rule-based parser by default for deterministic queries:
       (object, color, intent, camera/location, time range, negation)
    2. Ollama llama3.2:1b only as a fallback when the rule parser cannot confidently
       parse the query.
    If Ollama or the model is unavailable:
       - Log a WARNING.
       - Expose the fallback state in /api/health.
       - Continue with the rule-based parser.
       - Never silently claim that an LLM parser was used.
    """
    global FALLBACK_ACTIVE, LAST_FALLBACK_REASON, LAST_PARSER_USED

    if config is None:
        config = load_config()

    # Step 1: Run deterministic RuleQueryParser first
    rule_parsed = RuleQueryParser.parse(raw_query)
    is_confident = RuleQueryParser.is_confident(rule_parsed, raw_query)
    parser_mode = getattr(config.query, "parser_mode", "rules_first")

    if parser_mode == "rules_only" or not use_llm or is_confident:
        LAST_PARSER_USED = "rules"
        FALLBACK_ACTIVE = False
        LAST_FALLBACK_REASON = None
        return rule_parsed

    # Step 2: Fallback to Ollama LLM parser for ambiguous or unconfident queries
    logger.info("Rule parser produced low confidence for '%s'. Attempting Ollama LLM fallback.", raw_query)
    if config.query.parser_model:
        parser = LLMQueryParser(
            ollama_url=config.query.ollama_url,
            model_name=config.query.parser_model,
            timeout_s=config.query.parser_timeout_s,
        )
        return parser.parse(raw_query)

    LAST_PARSER_USED = "rules"
    return rule_parsed
