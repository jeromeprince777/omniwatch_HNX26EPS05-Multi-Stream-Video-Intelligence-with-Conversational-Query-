import os
from pathlib import Path

DIR = Path(r"c:\Users\ffgam\OneDrive\Desktop\omniwatch_pack\docs\screenshots")
old_webp = DIR / "09_interactive_demo_session.webp"
new_webp = DIR / "demo_surveillance_session.webp"

if old_webp.exists():
    if new_webp.exists():
        new_webp.unlink()
    old_webp.rename(new_webp)
    print("Renamed to demo_surveillance_session.webp")
