"""Reproduce the CPU verification suite. Run prepare_data.py once first."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
commands=[['-m','unittest','discover','-s',str(ROOT),'-p','test_*.py','-v'],
          [str(ROOT/'run_classification_round2.py')],[str(ROOT/'run_retrieval_round2.py')],
          [str(ROOT/'run_heldout_retrieval.py')],[str(ROOT/'run_matched_feedback.py')],
          [str(ROOT/'run_gait_integration.py')]]
if '--references' in sys.argv:
    commands += [[str(ROOT/'reference_checks/compare_upstreams.py')],[str(ROOT/'reference_checks/compare_biased.py')]]
for args in commands:
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*args],check=True,cwd=ROOT)
