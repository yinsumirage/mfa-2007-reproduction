"""Fetch and verify local data. No image redistribution is part of this code."""
from pathlib import Path
import hashlib,json,subprocess
from sklearn.datasets import fetch_olivetti_faces
ROOT=Path(__file__).resolve().parent
fetch_olivetti_faces(data_home=str(ROOT/'.data-cache'),shuffle=False)
manifest=json.loads((ROOT.parent/'results/round2/corel1k-source.json').read_text())
repo=ROOT/'.data-cache/corel1k-mirror'
if not repo.exists():
    subprocess.run(['git','clone','--no-checkout',manifest['source'],str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',manifest['commit']],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if commit!=manifest['commit']:raise RuntimeError('Corel mirror commit differs from the recorded data version')
for item in manifest['files']:
    f=repo/'img'/item['name']
    if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Data hash mismatch: '+item['name'])
print('Olivetti downloaded/verified by sklearn; all 1,000 Corel mirror file hashes match.')
