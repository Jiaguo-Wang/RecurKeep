"""Load bundled synthetic logical fixtures; these do not represent real GPU state or certification results."""
import json
from pathlib import Path
from experiments.epoch_io import load_epoch, with_budget
ROOT=Path(__file__).resolve().parents[1]


def load(group=1,k=1):
    """Reuse the synthetic branching matrix; each test creates its own simulated physical identities."""
    if group!=1:raise ValueError('本最小夹具只支持 group=1')
    data=ROOT/'examples/data'
    manifest=json.loads((data/'manifest.json').read_text())
    case=next(x for x in manifest['cases'] if x['name']=='demo')
    return with_budget(load_epoch(data/case['file'],case['epoch_sha256']),k)
