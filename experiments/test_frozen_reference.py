"""Historical regression tests requiring standalone frozen archives: python3 -m unittest experiments.test_frozen_reference -v."""
import csv
import json
from pathlib import Path
import unittest

from experiments.epoch_io import load_epoch, with_budget
from recurkeep.optimizer import GlobalOptimizer
from recurkeep.recovery_model import RecoveryCostModel
from recurkeep.strict_control import validate_plan

ROOT = Path(__file__).resolve().parents[1]


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = GlobalOptimizer(RecoveryCostModel())

    def test_strict20_and_ck3_reference(self):
        data=ROOT/'artifacts/frozen_cpu_snapshots'
        manifest=json.loads((data/'manifest.json').read_text())
        with (data/'solver_dispatch_cpu_recheck.csv').open() as handle:
            references={(r['condition'].lower().replace('b5-c4','b5c4').replace('c/k3','ck3'),int(r['K'])):r
                        for r in csv.DictReader(handle)}
        count=0
        for case in manifest['cases']:
            base=load_epoch(data/case['file'],case['epoch_sha256'])
            for budget in case['budgets']:
                with self.subTest(case=case['name'],K=budget):
                    epoch=with_budget(base,budget)
                    plan=self.optimizer.plan_strict(epoch);validate_plan(epoch,plan)
                    row=references[(case['name'],budget)]
                    self.assertEqual(set(plan.selected_ids),set(json.loads(row['selected'])))
                    self.assertAlmostEqual(plan.recovery_cost_ms,float(row['J_ms']),places=9)
                    self.assertEqual(self.optimizer.last_strict_decision['solver'],row['actual_solver'])
                    count+=1
        self.assertEqual(count,21)


if __name__ == '__main__':
    unittest.main()
