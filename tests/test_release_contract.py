"""Standard-library regression tests for synthetic inputs, strict contracts, and solver dispatch."""
from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.epoch_io import load_epoch, with_budget
from recurkeep.optimizer import GlobalOptimizer, _exact_admission
from recurkeep.recovery_model import RecoveryCostModel
from recurkeep.strict_control import validate_plan
from recurkeep.strict_epoch import _digest
from tests.fixtures import load

ROOT=Path(__file__).resolve().parents[1]


def reseal(epoch, **changes):
    """Recompute only the outer digest for negative cases so internal validation can independently detect errors."""
    fields={name:getattr(epoch,name) for name in epoch.__dataclass_fields__ if name!='content_digest'}
    fields.update(changes)
    payload={name:tuple(asdict(x) for x in value) if name in ('candidates','pendings','edges') else value
             for name,value in fields.items()}
    return replace(epoch,**changes,content_digest=_digest(payload))


class ReleaseContractTests(unittest.TestCase):
    def setUp(self):
        self.epoch=load(k=2)
        self.optimizer=GlobalOptimizer(RecoveryCostModel())

    def test_synthetic_branch_plan(self):
        plan=self.optimizer.plan_strict(self.epoch)
        validate_plan(self.epoch,plan)
        self.assertEqual(set(plan.selected_ids),{'branch-a-deep','branch-b'})
        self.assertEqual(self.epoch.frontiers(plan.selected_ids),(6,4))
        self.assertEqual({p.pending_id:p.G for p in plan.pending},{'pending-a':2,'pending-b':4})

    def test_unknown_rejected(self):
        edge=replace(self.epoch.edges[0],status='UNKNOWN')
        with self.assertRaises(ValueError):reseal(self.epoch,edges=(edge,)+self.epoch.edges[1:])

    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):reseal(self.epoch,edges=self.epoch.edges[1:])

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):reseal(self.epoch,candidates=self.epoch.candidates+(self.epoch.candidates[0],))

    def test_corrupt_tokens_rejected(self):
        candidate=replace(self.epoch.candidates[0],token_ids=(999999,))
        with self.assertRaises(ValueError):reseal(self.epoch,candidates=(candidate,)+self.epoch.candidates[1:])

    def test_missing_components_rejected(self):
        candidate=self.epoch.candidates[0]
        broken=replace(candidate,candidate=replace(candidate.candidate,recurrent_resident=False))
        with self.assertRaises(ValueError):reseal(self.epoch,candidates=(broken,)+self.epoch.candidates[1:])

    def test_unequal_cost_rejected(self):
        candidate=self.epoch.candidates[0]
        broken=replace(candidate,candidate=replace(candidate.candidate,memory_bytes=1))
        with self.assertRaises(ValueError):reseal(self.epoch,candidates=(broken,)+self.epoch.candidates[1:])

    def test_epoch_identity_rejected(self):
        with self.assertRaises(ValueError):reseal(self.epoch,epoch_id='过期边界')

    def test_bad_budgets_rejected(self):
        for value in [-1,1.5,True]:
            with self.subTest(K=value),self.assertRaises(ValueError):with_budget(self.epoch,value)

    def test_plan_wrong_budget_rejected(self):
        plan=self.optimizer.plan_strict(self.epoch)
        with self.assertRaises(ValueError):validate_plan(self.epoch,replace(plan,K=1))

    def test_exact_error_never_falls_back(self):
        with patch.object(self.optimizer,'_enumerate_exact_subsets',side_effect=RuntimeError('模拟 Exact 异常')):
            with self.assertRaises(RuntimeError):self.optimizer.plan_strict(self.epoch)
        self.assertIsNone(self.optimizer.last_strict_decision)

    def test_exact_domain(self):
        for n,m,k,expected in [(12,4,4,True),(13,4,4,False),(12,5,4,False),(12,4,5,False),(16,4,3,False)]:
            with self.subTest(n=n,m=m,k=k):self.assertEqual(_exact_admission(n,m,k)[0],expected)

    def test_corrupt_archive_hash_rejected(self):
        path=ROOT/'examples/data/demo.json'
        with self.assertRaises(ValueError):load_epoch(path,'0'*64)

    def test_model_boundary_and_monotonic_domain(self):
        model=RecoveryCostModel()
        self.assertEqual(model.estimate(0,131072),0)
        # For the linear derivative with g <= tau <= 128, the minimum is at a boundary; frozen coefficients satisfy theta2 + 2theta3 > 0.
        coefficients=model.metadata
        self.assertGreater(coefficients.coefficient_b+2*coefficients.coefficient_c,0)
        self.assertGreater(coefficients.coefficient_a,0)
        for g,t in [(-1,1),(2,1),(1,131073)]:
            with self.assertRaises(ValueError):model.estimate(g,t)


if __name__=='__main__':unittest.main()
