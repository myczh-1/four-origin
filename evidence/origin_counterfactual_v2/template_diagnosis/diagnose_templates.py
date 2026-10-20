"""Frozen-checkpoint 2x2 template diagnosis; no training or checkpoint writes."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
import run_experiment as exp

TEMPLATES = {
    'number': ('Project {e} has a connection limit of {v}.', 'For project {e}, the connection cap equals {v}.'),
    'name': ('The owner of project {e} is {v}.', 'Project {e} belongs to {v}.'),
    'configuration': ('Project {e} uses the {v} cache policy.', 'The cache policy selected for {e} is {v}.'),
    'path': ('Project {e} stores its log at {v}.', 'The log location for project {e} is {v}.'),
    'date': ('Project {e} has a release date of {v}.', 'The release of project {e} is scheduled for {v}.'),
    'boolean': ('Project {e} has audit logging set to {v}.', 'For project {e}, audit logging is {v}.')}
QUESTIONS = ['Does the reference material confirm this statement?',
             'Is this statement confirmed by the reference material?']


def stats(rows):
    pairs = defaultdict(list)
    for r in rows:
        pairs[r['pair_id']].append(r)
    assert all(len(p) == 2 for p in pairs.values())
    selfs = [r for r in rows if r['origin_id'] == exp.SELF]
    externals = [r for r in rows if r['origin_id'] == exp.EXTERNAL]
    return {'samples': len(rows), 'pairs': len(pairs),
            'correct_pairs': sum(all(r['raw'] == r['answer'] for r in p) for p in pairs.values()),
            'pair_accuracy': sum(all(r['raw'] == r['answer'] for r in p) for p in pairs.values()) / len(pairs),
            'sample_accuracy': sum(r['raw'] == r['answer'] for r in rows) / len(rows),
            'self_errors': sum(r['raw'] != r['answer'] for r in selfs),
            'external_errors': sum(r['raw'] != r['answer'] for r in externals),
            'invalid_outputs': sum(r['raw'] not in ('YES', 'NO') for r in rows),
            'mean_self_logp_no_minus_yes': sum(r['logp_no'] - r['logp_yes'] for r in selfs) / len(selfs),
            'mean_self_probability_no': sum(math.exp(r['logp_no']) for r in selfs) / len(selfs)}


def main():
    protected = {str(p.relative_to(V2)): exp.sha(p) for p in V2.rglob('*')
                 if p.is_file() and HERE not in p.parents and '__pycache__' not in p.parts}
    raw = exp.read_jsonl(V2 / 'data/cross_template.jsonl')
    originals = [r for r in exp.read_jsonl(V2 / 'predictions.jsonl')
                 if r['variant'] == 'origin' and r['condition'] == 'normal' and r['split'] == 'cross_template']
    tok, model = exp.tokenizer(), exp.load_model('origin')
    rows_by_condition, outputs, summaries = {}, {}, {}
    for fi, qi in [(0,0), (1,0), (0,1), (1,1)]:
        name = f'F{fi}_Q{qi}'
        rows = []
        for r in raw:
            cross = TEMPLATES[r['kind']][1].replace('{e}', r['entity'])
            prefix, suffix = cross.split('{v}')
            assert r['fact'].startswith(prefix) and r['fact'].endswith(suffix)
            value = r['fact'][len(prefix):-len(suffix)]
            assert TEMPLATES[r['kind']][1].format(e=r['entity'], v=value) == r['fact']
            assert r['question'] == QUESTIONS[1]
            rows.append({**r, 'fact': TEMPLATES[r['kind']][fi].format(e=r['entity'], v=value),
                         'question': QUESTIONS[qi], 'value': value, 'factor_condition': name})
        for a,b in zip(rows[::2],rows[1::2]):
            ea,eb=exp.encode(tok,a),exp.encode(tok,b)
            assert ea['ids'] == eb['ids'] and ea['fact_mask'] == eb['fact_mask']
            assert all((x!=y)==mask for x,y,mask in zip(ea['origins'],eb['origins'],ea['fact_mask']))
        rows_by_condition[name] = rows
        _, predictions = exp.evaluate_rows(model,tok,rows,variant='origin',split='cross_template_factorial')
        outputs[name] = predictions
        summaries[name] = {'overall': stats(predictions),
                           'by_kind': {kind: stats([r for r in predictions if r['kind']==kind]) for kind in exp.KINDS},
                           'configuration_by_value': {v: stats([r for r in predictions if r['kind']=='configuration' and r['value']==v])
                                                      for v in sorted({r['value'] for r in predictions if r['kind']=='configuration'})}}
        print(name,json.dumps(summaries[name]),flush=True)
        (HERE/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for records in outputs.values() for r in records))
    # Verify that the factorial interventions change neither content identity nor source labels.
    for i, r in enumerate(raw):
        for name, rows in rows_by_condition.items():
            assert (rows[i]['pair_id'],rows[i]['entity'],rows[i]['origin_id'],rows[i]['answer']) == (r['pair_id'],r['entity'],r['origin_id'],r['answer'])
            assert rows[i]['value'] == rows_by_condition['F0_Q0'][i]['value']
        assert rows_by_condition['F0_Q0'][i]['fact'] == rows_by_condition['F0_Q1'][i]['fact']
        assert rows_by_condition['F1_Q0'][i]['fact'] == rows_by_condition['F1_Q1'][i]['fact']
        assert rows_by_condition['F0_Q0'][i]['question'] == rows_by_condition['F1_Q0'][i]['question']
        assert rows_by_condition['F0_Q1'][i]['question'] == rows_by_condition['F1_Q1'][i]['question']
    reproduced = outputs['F1_Q1']
    assert [(r['pair_id'],r['origin_id'],r['raw']) for r in reproduced] == [(r['pair_id'],r['origin_id'],r['raw']) for r in originals]
    reproduction_delta = max(abs(a[k]-b[k]) for a,b in zip(reproduced,originals) for k in ['logp_yes','logp_no'])
    contrasts = {}
    for group in ['overall'] + exp.KINDS:
        group_stats = {name: s['overall'] if group=='overall' else s['by_kind'][group] for name,s in summaries.items()}
        contrasts[group] = {}
        for metric in ['pair_accuracy','mean_self_logp_no_minus_yes']:
            v = {name:s[metric] for name,s in group_stats.items()}
            contrasts[group][metric] = {'fact_effect_with_old_question':v['F1_Q0']-v['F0_Q0'],
                'fact_effect_with_new_question':v['F1_Q1']-v['F0_Q1'],
                'question_effect_with_old_fact':v['F0_Q1']-v['F0_Q0'],
                'question_effect_with_new_fact':v['F1_Q1']-v['F1_Q0'],
                'interaction':v['F1_Q1']-v['F1_Q0']-v['F0_Q1']+v['F0_Q0']}
    error_transitions=[]
    for i,r in enumerate(raw):
        if r['origin_id'] != exp.SELF: continue
        error_transitions.append({'pair_id':r['pair_id'],'kind':r['kind'],'value':rows_by_condition['F0_Q0'][i]['value'],
            'answers':{name:records[i]['raw'] for name,records in outputs.items()},
            'no_minus_yes_logp':{name:records[i]['logp_no']-records[i]['logp_yes'] for name,records in outputs.items()}})
    repairs={}
    for group in ['overall','configuration']:
        old_failures=[r for r in error_transitions if r['answers']['F1_Q1']!='NO' and (group=='overall' or r['kind']==group)]
        repairs[group]={'original_self_errors':len(old_failures),
            'fixed_by_fact_only':sum(r['answers']['F0_Q1']=='NO' for r in old_failures),
            'fixed_by_question_only':sum(r['answers']['F1_Q0']=='NO' for r in old_failures),
            'fixed_by_both':sum(r['answers']['F0_Q0']=='NO' for r in old_failures),
            'error_pattern_counts':dict(Counter('/'.join(r['answers'][c] for c in ['F0_Q0','F1_Q0','F0_Q1','F1_Q1']) for r in old_failures))}
    unchanged=all(exp.sha(V2/p)==h for p,h in protected.items())
    assert unchanged
    result={'design':'Frozen 2x2 fact-template by question-template, same 200 facts/400 records from cross-template test.',
            'training_performed':False,'model':exp.MODEL,'revision':exp.REVISION,'batch_size':16,
            'fact_templates':TEMPLATES,'questions':QUESTIONS,'summaries':summaries,'contrasts':contrasts,
            'original_failure_repairs':repairs,'self_sample_transitions':error_transitions,
            'original_run_answers_reproduced':True,'original_run_max_logprob_delta':reproduction_delta,
            'all_previous_artifacts_unchanged':unchanged,'protected_sha256':protected,
            'script_sha256':exp.sha(Path(__file__)),
            'scope':'Post-hoc diagnosis of the observed test failures, not a new generalization estimate. Template changes include token identities, length and word order.'}
    exp.save_json(HERE/'results.json',result)
    print('CONTRASTS',json.dumps(contrasts,indent=2),flush=True)
    print('REPAIRS',json.dumps(repairs,indent=2),flush=True)


if __name__=='__main__':
    main()
