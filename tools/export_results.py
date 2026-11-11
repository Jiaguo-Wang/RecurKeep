"""Export existing local experiment evidence without running new experiments."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
TABLES = {}
SOURCES = {}


def source(path, member=None):
    raw = (ROOT / path).read_bytes()
    if member:
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            raw = archive.extractfile(member).read()
    label = path + ('::' + member if member else '')
    SOURCES[label] = hashlib.sha256(raw).hexdigest()
    return raw.decode(), label


def rows(path, member=None):
    text, label = source(path, member)
    return list(csv.DictReader(io.StringIO(text))), label


def emit(name, records):
    assert records, name
    fields = list(dict.fromkeys(key for row in records for key in row))
    with (OUT / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    TABLES[name] = len(records)


def metadata(group, case, budget, policy, label, row):
    case = case.lower()
    return dict(experiment_group=group, condition_id=f'{case}/K{budget}',
                case=case, budget_k=budget, policy=policy, source=label, source_row=row)


def main():
    OUT.mkdir(exist_ok=True)
    historical, hs = rows('artifacts/frozen_cpu_snapshots/historical_policy_results.csv')
    archived, ars = rows('archives/frozen_cpu_snapshots.tar.gz',
                        'frozen_cpu_snapshots/historical_policy_results.csv')
    current, cs = rows('validation/recurkeep_rename/cpu_comparison.csv')
    assert len(historical) == len(archived) == 140
    assert len(current) == 63
    # Compare by identity rather than assuming the two files have identical ordering.
    identity = lambda r: (r['case'], r['K'], r['policy'])
    frozen = {identity(r): r for r in archived}
    assert len(frozen) == len(archived)
    for row in historical:
        old = frozen[identity(row)]
        for key in row:
            if not key.startswith('solve_'):
                assert row[key] == old[key], (identity(row), key)

    allocation, frontiers, quality, timing = [], [], [], []
    exact = {(r['case'].lower(), r['K']): float(r['J_ms'])
             for r in historical if r['policy'] == 'Exact'}
    for i, r in enumerate(historical, 2):
        meta = metadata('historical_strict20', r['case'], r['K'], r['policy'], hs, i)
        selected = json.loads(r['selected'])
        allocation.append(dict(meta, selected_ids_json=r['selected'], selected_count=len(selected),
                               J_model_ms=r['J_ms'], evidence_type='historical_cpu_model'))
        pending = json.loads(r['per_pending'])
        assert abs(sum(p['cost_ms'] for p in pending) - float(r['J_ms'])) < 1e-6
        for p in pending:
            assert p['T'] - p['E'] == p['G']
            frontiers.append(dict(meta, pending_id=p['continuation_id'], T_tokens=p['T'],
                                  E_tokens=p['E'], G_tokens=p['G'], cost_model_ms=p['cost_ms'],
                                  evidence_type='modeled_frontier'))
        gap = float(r['J_ms']) - exact[(r['case'].lower(), r['K'])]
        assert abs(gap - float(r['gap_exact_ms'])) < 1e-6
        quality.append(dict(meta, J_model_ms=r['J_ms'], exact_J_model_ms=exact[(r['case'].lower(),r['K'])],
                            gap_exact_ms=gap, matches_exact_objective=abs(gap)<1e-9,
                            baseline_status='exact_available'))

    for group, records, label, timing_kind in [
        ('historical_strict20_local', historical, hs, 'local_calibrated_per_documentation'),
        ('historical_strict20_archive', archived, ars, 'archived_timing_as_recorded')]:
        for i, r in enumerate(records, 2):
            timing.append(dict(metadata(group,r['case'],r['K'],r['policy'],label,i),
                               timing_kind=timing_kind, solve_mean_ms=r['solve_mean_ms'],
                               solve_median_ms=r['solve_median_ms'], solve_min_ms=r['solve_min_ms'],
                               solve_max_ms=r['solve_max_ms']))

    for i, r in enumerate(current, 2):
        meta = metadata('rename_cpu21',r['snapshot'],r['K'],r['policy'],cs,i)
        selected = json.loads(r['selected'])
        assert len(selected) <= int(r['K'])
        allocation.append(dict(meta, selected_ids_json=r['selected'], selected_count=len(selected),
                               candidate_count=r['candidate_count'], J_model_ms=r['J_ms'],
                               evidence_type='historical_cpu_model', solver=r['solver']))
        epoch_path = f"artifacts/frozen_cpu_snapshots/{r['snapshot']}.json"
        epoch = json.loads(source(epoch_path)[0])
        es, gs = json.loads(r['E']), json.loads(r['G'])
        assert len(es) == len(gs) == len(epoch['pendings'])
        for p, e, g in zip(epoch['pendings'], es, gs):
            assert p['target'] - e == g
            frontiers.append(dict(meta, pending_id=p['pending_id'], T_tokens=p['target'],
                                  E_tokens=e,G_tokens=g,evidence_type='modeled_frontier'))
        baseline = exact.get((r['snapshot'],r['K']))
        # The shared B5-C4 case uses different punctuation in the two exports.
        if r['snapshot'] == 'b5c4': baseline = exact.get(('b5-c4',r['K']))
        gap = float(r['J_ms']) - baseline if baseline is not None else ''
        quality.append(dict(meta,J_model_ms=r['J_ms'],exact_J_model_ms=baseline if baseline is not None else '',
                            gap_exact_ms=gap,matches_exact_objective=abs(gap)<1e-9 if baseline is not None else '',
                            baseline_status='exact_available' if baseline is not None else 'exact_unavailable'))
        timing.append(dict(meta,timing_kind='historical_single_run',solve_ms=r['solve_ms']))

    profile, ps = rows('recurkeep/profiles/replay_cost.csv')
    fit_text, fs = source('recurkeep/profiles/fit_metrics.json')
    fit = json.loads(fit_text)['group_median_ols']
    gpu, costs = [], []
    for i, r in enumerate(profile, 2):
        tokens = int(r['expected_replay_tokens'])
        meta = dict(experiment_group='historical_replay_calibration',condition_id=f'replay/{tokens}',
                    source=ps,source_row=i)
        gpu.append(dict(meta,**r,evidence_type='historical_profile_aggregate',
                        timing_scope='recovery_latency_as_recorded_not_verified_cuda_prefill',
                        raw_repetitions_available=False))
        predicted = fit['intercept_ms'] + fit['slope_ms_per_token'] * tokens
        measured = float(r['recovery_latency_median_ms'])
        costs.append(dict(meta,model='historical_group_median_ols',replay_tokens=tokens,
                          observed_recovery_median_ms=measured,predicted_recovery_ms=predicted,
                          residual_observed_minus_predicted_ms=measured-predicted,
                          absolute_error_ms=abs(measured-predicted),fit_source=fs,
                          intercept_ms=fit['intercept_ms'],slope_ms_per_token=fit['slope_ms_per_token'],
                          evaluation_scope='in_sample_fit_not_formal_position_aware_model'))

    member = 'runtime_reference/evaluation/serving_control_gpu_gate_20261102/correctness_evidence.json'
    raw, vs = source('archives/runtime_reference.tar.gz',member)
    evidence = json.loads(raw)
    validation = []
    for ai, arm in enumerate(evidence['arms']):
        common = dict(experiment_group='historical_runtime_ck3',condition_id='ck3/K3',
                      arm=arm['arm'],mode=arm['mode'],source=vs,
                      provenance=arm['provenance'],evidence_type='archived_not_rerun')
        validation.append(dict(common,check='arm_status',value=arm['status'],
                               source_pointer=f'/arms/{ai}/status'))
        for key, value in arm.items():
            if isinstance(value,bool) or key in ('completed_evictions','free_before','free_after','resident_before','resident_after'):
                validation.append(dict(common,check=key,value=value,source_pointer=f'/arms/{ai}/{key}'))
        for pi, p in enumerate(arm['pending_evidence']):
            validation.append(dict(common,pending_id=p['pending_id'],check='output_tokens_equal_cold',
                                   value=p['actual_output_ids']==p['cold_output_ids'],
                                   source_pointer=f'/arms/{ai}/pending_evidence/{pi}',
                                   derivation='actual_output_ids == cold_output_ids'))
            n = p['native']
            e,g = n['executable_prefix'],n['replay_gap']
            frontiers.append(dict(experiment_group='historical_runtime_ck3',condition_id='ck3/K3',
                                  case='ck3',budget_k=3,policy='archived_selected_plan',arm=arm['arm'],
                                  pending_id=p['pending_id'],T_tokens=e+g,E_tokens=e,G_tokens=g,
                                  input_token_count=n['input_token_count'],physical_fa_hit_tokens=n['physical_fa_hit'],
                                  evidence_type='archived_native_observation',provenance=arm['provenance'],
                                  source=vs,source_pointer=f'/arms/{ai}/pending_evidence/{pi}/native'))
    for key,value in evidence['cross_lifecycle'].items():
        validation.append(dict(experiment_group='historical_runtime_ck3',condition_id='ck3/K3',
                               arm='cross_lifecycle',check=key,value=value,source=vs,
                               source_pointer=f'/cross_lifecycle/{key}',evidence_type='archived_not_rerun',
                               provenance='PROVENANCE_UNVERIFIED'))
    for name, records in [('gpu_recovery',gpu),('recovery_frontiers',frontiers),('cost_model',costs),
                          ('checkpoint_allocation',allocation),('solver_quality',quality),
                          ('solver_runtime',timing),('runtime_validation',validation)]:
        emit(name,records)
    receipt = {'source_sha256':SOURCES,'output_rows':TABLES,
               'checks':['historical non-timing fields match archive','per-pending costs sum to J',
                         'T-E=G','historical gaps match Exact','selected counts respect current budgets'],
               'scope':'Exported existing evidence only; no new CPU/GPU experiment runs.'}
    (OUT/'sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(TABLES,indent=2))


if __name__ == '__main__':
    main()
