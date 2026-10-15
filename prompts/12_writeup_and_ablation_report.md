# 12 — Write-up and ablation report
**Read:** docs/07_WRITEUP_TEMPLATE.md, docs/02, `eval/results/**`.

**Task:**
1. Run `eval/make_tables.py` and paste its output into `WRITEUP.md` following the template headings. Every number must come from `eval/results/*.json`; cite the run id next to each table.
2. Describe only what exists in the repository. For each claim in section 8 ("what is new"), name the ablation row that supports it. Remove any claim without evidence.
3. Failure analysis: pick 3-5 failed or wrong test queries from `per_query.jsonl`, show the thumbnail/clip ids and a one-line cause each (detector miss, color confusion, time offset, referent wrong, etc.).
4. State assumptions and limits honestly: query-set size, own labeling, dataset, hardware, KB scope per site, cancel path, bonus items not built.
5. Reproduce section with the exact three commands verified by running them from a clean checkout.
6. Length: about two pages. No marketing language.

**Acceptance:** a script `scripts/check_writeup_numbers.py` that extracts every number in the tables from `WRITEUP.md` and confirms it matches `eval/results/`. Show it passing.
