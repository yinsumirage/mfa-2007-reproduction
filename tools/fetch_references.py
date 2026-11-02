"""Fetch optional reference source at recorded commits; do not vendor it."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'code/upstream-manifest.json').read_text())
for entry in manifest['repositories']:
    destination = ROOT / 'code' / entry['directory']
    if destination.exists():
        head = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        if head != entry['commit']:
            raise SystemExit(f'Reference checkout differs from the pinned commit: {destination.name}')
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', entry['remote'], str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', entry['commit']], check=True)
    print(entry['directory'], entry['commit'])
