"""Restore frozen logical epochs only; do not bind historical physical identities to the current runtime."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from recurkeep.state_catalog import CheckpointCandidate
from recurkeep.strict_epoch import (StrictAllocationEpoch, StrictCandidate,
                                    StrictPending, StrictEdge, _digest)


def load_epoch(path, expected_sha256):
    """Verify the archived byte digest, then restore immutable fields and enforce the original strict contract."""
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('冻结 Epoch 文件摘要不一致')
    data=json.loads(raw)
    candidates=[]
    for entry in data['candidates']:
        item=dict(entry);core=dict(item['candidate'])
        core['lineage_path']=tuple(core['lineage_path'])
        item['candidate']=CheckpointCandidate(**core)
        item['token_ids']=tuple(item['token_ids'])
        candidates.append(StrictCandidate(**item))
    data['candidates']=tuple(candidates)
    data['pendings']=tuple(StrictPending(**dict(p,token_ids=tuple(p['token_ids']))) for p in data['pendings'])
    data['edges']=tuple(StrictEdge(**dict(e,conditions=tuple(tuple(x) for x in e['conditions']))) for e in data['edges'])
    return StrictAllocationEpoch(**data)


def with_budget(epoch, budget):
    """Derive only predefined budget conditions, preserving candidates, inputs, and strict relationships."""
    payload=asdict(epoch)
    payload.pop('content_digest')
    payload['K']=budget
    return replace(epoch,K=budget,content_digest=_digest(payload))
