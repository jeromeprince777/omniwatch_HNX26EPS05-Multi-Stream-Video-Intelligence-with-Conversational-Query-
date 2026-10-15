# Reset only the demo site rows and caches; refuse to touch eval/ (PowerShell)
param (
    [string]$TargetPath = "data/cache"
)

$ErrorActionPreference = "Stop"

if ($TargetPath -match "eval") {
    Write-Error "ERROR: Refusing to reset or touch eval/ directory! (AGENTS.md contract)"
    exit 1
}

Write-Host "[OmniWatch] Resetting demo cache and demo database..." -ForegroundColor Yellow

$pythonCmd = @"
import shutil, os, sqlite3
cache_dir = 'data/cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)
db_path = 'data/omniwatch.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    for tbl in ['observations', 'tracklets', 'cameras', 'sites', 'clarification_log', 'pending_queries', 'query_log']:
        try:
            conn.execute(f'DELETE FROM {tbl}')
        except Exception:
            pass
    conn.commit()
    conn.close()
print('Demo cache and tables successfully reset.')
"@

python -c $pythonCmd
