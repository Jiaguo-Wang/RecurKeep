"""Compare the core hybrid solver, original marginal greedy solver, and independent ranking on the same frozen matrix."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

from recurkeep.optimizer import GlobalOptimizer
from recurkeep.recovery_model import RecoveryCostModel
from recurkeep.strict_control import validate_plan
from .epoch_io import load_epoch, with_budget


def run(data, output):
    """Check twenty conditions across five snapshots and the C/K3 fallback, reporting only the modeled objective."""
    manifest=json.loads((data/'manifest.json').read_text())
    references={(r['condition'].lower().replace('b5-c4','b5c4').replace('c/k3','ck3'),int(r['K'])):r
                for r in csv.DictReader((data/'solver_dispatch_cpu_recheck.csv').open())}
    model=RecoveryCostModel();rows=[]
    for case in manifest['cases']:
        base=load_epoch(data/case['file'],case['epoch_sha256'])
        for budget in case['budgets']:
            epoch=with_budget(base,budget)
            optimizer=GlobalOptimizer(model)
            plan=optimizer.plan_strict(epoch)
            validate_plan(epoch,plan)
            reference=references[(case['name'],budget)]
            expected=json.loads(reference['selected'])
            if set(plan.selected_ids)!=set(expected) or abs(plan.recovery_cost_ms-float(reference['J_ms']))>1e-9:
                raise AssertionError('与历史主库复核结果不一致：'+case['name'])
            def score(ids):
                return sum(model.estimate(p.target-e,p.target) for p,e in zip(epoch.pendings,epoch.frontiers(ids)))
            candidates=tuple(c.candidate for c in epoch.candidates)
            started=time.perf_counter_ns()
            greedy=optimizer._select_candidates(candidates,budget*candidates[0].memory_bytes,
                    lambda selected:score(tuple(c.checkpoint_id for c in selected)))
            greedy_ms=(time.perf_counter_ns()-started)/1e6
            started=time.perf_counter_ns()
            standalone=tuple(c.checkpoint_id for c in sorted(candidates,
                        key=lambda c:(-(score(())-score((c.checkpoint_id,))),c.checkpoint_id))[:budget])
            standalone_ms=(time.perf_counter_ns()-started)/1e6
            records=[('RecurKeep',plan.selected_ids,optimizer.last_strict_decision['solver'],
                      optimizer.last_strict_decision['pure_solve_ms']),
                     ('Marginal Greedy',tuple(c.checkpoint_id for c in greedy.selected),'Marginal Greedy',greedy_ms),
                     ('Independent Top-K',standalone,'Independent Top-K',standalone_ms)]
            for policy,ids,solver,elapsed in records:
                frontiers=epoch.frontiers(ids)
                rows.append({'snapshot':case['name'],'K':budget,'candidate_count':len(candidates),
                    'pending_count':len(epoch.pendings),'policy':policy,'solver':solver,
                    'selected':json.dumps(list(ids)),'J_ms':score(ids),
                    'E':json.dumps(list(frontiers)),
                    'G':json.dumps([p.target-e for p,e in zip(epoch.pendings,frontiers)]),
                    'solve_ms':elapsed,'frozen_reference_match':'PASS' if policy=='RecurKeep' else '',
                    'scope':'冻结 Epoch 的 CPU 模型化评价'})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'PASS','conditions':21,'policy_rows':len(rows),'output':str(output)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('artifacts/frozen_cpu_snapshots'))
    parser.add_argument('--output',type=Path,default=Path('results/cpu_comparison.csv'))
    args=parser.parse_args();run(args.data,args.output)
