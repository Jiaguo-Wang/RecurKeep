"""Run CPU-only planning with the bundled synthetic example by default, without executing evictions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.epoch_io import load_epoch, with_budget
from recurkeep.optimizer import GlobalOptimizer
from recurkeep.recovery_model import RecoveryCostModel
from recurkeep.strict_control import validate_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,help='输入目录；默认 demo 使用随仓库数据，其他快照使用冻结归档')
    parser.add_argument('--snapshot',default='demo')
    parser.add_argument('--budget',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    data=args.data or (root/'examples/data' if args.snapshot=='demo' else root/'artifacts/frozen_cpu_snapshots')
    if not (data/'manifest.json').is_file():
        parser.error('找不到输入清单；最小示例请直接运行 python3 examples/plan.py，冻结实验请先按 docs/experiments.md 准备归档')
    manifest=json.loads((data/'manifest.json').read_text())
    case=next((c for c in manifest['cases'] if c['name']==args.snapshot),None)
    if case is None:
        parser.error('输入清单中没有此快照：'+args.snapshot)
    if args.budget not in case['budgets']:
        parser.error('输入清单未声明此预算条件')
    epoch=with_budget(load_epoch(data/case['file'],case['epoch_sha256']),args.budget)
    optimizer=GlobalOptimizer(RecoveryCostModel());plan=optimizer.plan_strict(epoch)
    validate_plan(epoch,plan)
    print(json.dumps({'solver':optimizer.last_strict_decision,'plan':asdict(plan),
                      'input_scope':manifest.get('scope','Archived frozen CPU snapshot'),
                      '说明':'仅 CPU 逻辑规划，不代表当前 GPU 驻留或实际恢复耗时'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
