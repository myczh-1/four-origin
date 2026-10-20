"""Strict origin counterfactual experiment. All writes stay in this directory."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SYSTEM, USER, EXTERNAL, SELF = range(4)
SEED = 2026
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DTYPE = torch.float16 if DEVICE.type == 'mps' else torch.float32
GEN_TOKENS = 6
KINDS = ['number', 'name', 'configuration', 'path', 'date', 'boolean']


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)


def encode(tok, row, tagged=False, answer=False, fact_origin=None):
    source = row['origin_id'] if fact_origin is None else fact_origin
    # Only the fact span changes origin in counterfactuals. No role tags in C inputs.
    tag = ('[EXTERNAL] ' if source == EXTERNAL else '[SELF] ') if tagged else ''
    segments = [(SYSTEM, 'Answer the question with exactly YES or NO.\n'),
                (source, tag + row['fact'] + '\n'),
                (USER, row['question'] + '\n'), (SELF, 'Answer:\n')]
    ids, origins, fact_mask = [], [], []
    for i, (origin, content) in enumerate(segments):
        part = tok.encode(content, add_special_tokens=False)
        ids.extend(part)
        origins.extend([origin] * len(part))
        fact_mask.extend([i == 1] * len(part))
    prompt_length = len(ids)
    if answer:
        target = tok.encode(row['answer'], add_special_tokens=False) + [tok.eos_token_id]
        ids.extend(target)
        origins.extend([SELF] * len(target))
        fact_mask.extend([False] * len(target))
    assert len(ids) <= 256
    return {'ids': ids, 'origins': origins, 'fact_mask': fact_mask,
            'labels': [-100] * prompt_length + ids[prompt_length:]}


def make_data():
    tok = tokenizer()
    (HERE / 'data').mkdir(exist_ok=True)
    rng = random.Random(271828)
    seen_entities, seen_facts = set(), set()
    report = {'splits': {}, 'pair_invariants': True, 'disjoint_facts_and_entities': True,
              'no_origin_text_hints': True, 'seed': 271828}
    templates = {
        'number': ('Project {e} has a connection limit of {v}.', 'For project {e}, the connection cap equals {v}.'),
        'name': ('The owner of project {e} is {v}.', 'Project {e} belongs to {v}.'),
        'configuration': ('Project {e} uses the {v} cache policy.', 'The cache policy selected for {e} is {v}.'),
        'path': ('Project {e} stores its log at {v}.', 'The log location for project {e} is {v}.'),
        'date': ('Project {e} has a release date of {v}.', 'The release of project {e} is scheduled for {v}.'),
        'boolean': ('Project {e} has audit logging set to {v}.', 'For project {e}, audit logging is {v}.')}
    for split, count in [('train', 2000), ('validation', 300), ('test', 500), ('cross_template', 200)]:
        rows = []
        for index in range(count):
            kind = KINDS[index % len(KINDS)]
            entity = 'p_' + f'{rng.getrandbits(48):012x}'
            assert entity not in seen_entities
            seen_entities.add(entity)
            values = {'number': str(rng.randrange(10, 10000)),
                      'name': 'owner_' + f'{rng.getrandbits(32):08x}',
                      'configuration': rng.choice(['fifo', 'lru', 'lfu', 'disabled', 'write-through', 'write-back']),
                      'path': '/srv/' + f'{rng.getrandbits(40):010x}' + '/events.log',
                      'date': f'{rng.randrange(2020, 2040)}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}',
                      'boolean': rng.choice(['true', 'false'])}
            fact = templates[kind][int(split == 'cross_template')].format(e=entity, v=values[kind])
            assert fact not in seen_facts
            seen_facts.add(fact)
            question = ('Is this statement confirmed by the reference material?' if split == 'cross_template'
                        else 'Does the reference material confirm this statement?')
            for source, label in [(EXTERNAL, 'YES'), (SELF, 'NO')]:
                rows.append({'pair_id': f'{split}-{index:04d}', 'kind': kind, 'entity': entity,
                             'fact': fact, 'question': question, 'origin_id': source, 'answer': label})
        lengths = []
        for i in range(0, len(rows), 2):
            a, b = rows[i:i+2]
            ea, eb = encode(tok, a), encode(tok, b)
            assert ea['ids'] == eb['ids']
            assert a['question'] == b['question'] and a['answer'] != b['answer']
            assert ea['fact_mask'] == eb['fact_mask']
            assert all((x != y) == mask for x, y, mask in zip(ea['origins'], eb['origins'], ea['fact_mask']))
            assert 'EXTERNAL' not in tok.decode(ea['ids']) and 'SELF' not in tok.decode(ea['ids'])
            # Both masks are all ones; position IDs are range(length); generation is shared.
            assert list(range(len(ea['ids']))) == list(range(len(eb['ids'])))
            assert [1] * len(ea['ids']) == [1] * len(eb['ids'])
            lengths.append(len(ea['ids']))
        path = HERE / 'data' / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        report['splits'][split] = {'pairs': count, 'samples': len(rows), 'sha256': sha(path),
                                  'max_prompt_tokens': max(lengths)}
    report['generation'] = {'greedy': True, 'max_new_tokens': GEN_TOKENS, 'use_cache': False}
    save_json(HERE / 'data_validation.json', report)
    print(json.dumps(report, indent=2), flush=True)


def batch(tok, rows, tagged=False, answer=False, sources=None):
    enc = [encode(tok, row, tagged, answer, None if sources is None else sources[i]) for i, row in enumerate(rows)]
    width = max(len(e['ids']) for e in enc)
    ids, origins, masks, labels = [], [], [], []
    for e in enc:
        pad = width - len(e['ids'])
        ids.append([tok.eos_token_id] * pad + e['ids'])
        origins.append([SYSTEM] * pad + e['origins'])
        masks.append([0] * pad + [1] * len(e['ids']))
        labels.append([-100] * pad + e['labels'])
    return {key: torch.tensor(value, device=DEVICE) for key, value in
            [('input_ids', ids), ('origin_ids', origins), ('attention_mask', masks), ('labels', labels)]}


class OriginModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.origin_embedding = nn.Embedding(4, lm.get_input_embeddings().embedding_dim)
        nn.init.zeros_(self.origin_embedding.weight)

    def forward(self, input_ids, origin_ids, attention_mask, labels=None, zero=False):
        embedded = self.lm.get_input_embeddings()(input_ids)
        if not zero:
            embedded = embedded + self.origin_embedding(origin_ids).to(embedded.dtype)
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        return self.lm(inputs_embeds=embedded, attention_mask=attention_mask,
                       position_ids=positions, labels=labels, use_cache=False)


def forward(model, tensors, labels=False, zero=False):
    if isinstance(model, OriginModel):
        return model(tensors['input_ids'], tensors['origin_ids'], tensors['attention_mask'],
                     tensors['labels'] if labels else None, zero=zero)
    return model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'],
                 position_ids=(tensors['attention_mask'].cumsum(-1) - 1).clamp_min(0),
                 labels=tensors['labels'] if labels else None, use_cache=False)


def load_model(variant, training=False):
    torch.manual_seed(SEED)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=DTYPE,
                                            low_cpu_mem_usage=True, trust_remote_code=False)
    if training:
        lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
                                          lora_dropout=0.0, bias='none', task_type='CAUSAL_LM'))
    elif variant != 'base':
        lm = PeftModel.from_pretrained(lm, HERE / 'checkpoints' / variant / 'lora')
    model = OriginModel(lm) if variant == 'origin' else lm
    if variant == 'origin' and not training:
        model.origin_embedding.load_state_dict(torch.load(HERE / 'checkpoints/origin/origin_embedding.pt',
                                                          map_location='cpu', weights_only=True))
    return model.to(DEVICE)


@torch.no_grad()
def predict(model, tok, rows, condition='normal', tagged=False, sources=None):
    model.eval()
    tensors = batch(tok, rows, tagged=tagged, sources=sources)
    first_logits = forward(model, tensors, zero=condition == 'zero').logits[:, -1].float()
    yes = tok.encode('YES', add_special_tokens=False)
    no = tok.encode('NO', add_special_tokens=False)
    assert len(yes) == len(no) == 1  # exact single-token candidate log probabilities
    probs = first_logits.log_softmax(-1)[:, [yes[0], no[0]]].cpu().tolist()
    generated = [[] for _ in rows]
    ended = [False] * len(rows)
    logits = first_logits
    for step in range(GEN_TOKENS):
        next_ids = logits.argmax(-1)
        values = next_ids.cpu().tolist()
        for i, token in enumerate(values):
            if not ended[i]:
                if token == tok.eos_token_id:
                    ended[i] = True
                else:
                    generated[i].append(token)
        if all(ended) or step == GEN_TOKENS - 1:
            break
        tensors['input_ids'] = torch.cat([tensors['input_ids'], next_ids[:, None]], dim=1)
        tensors['origin_ids'] = torch.cat([tensors['origin_ids'], torch.full_like(next_ids[:, None], SELF)], dim=1)
        tensors['attention_mask'] = torch.cat([tensors['attention_mask'], torch.ones_like(next_ids[:, None])], dim=1)
        logits = forward(model, tensors, zero=condition == 'zero').logits[:, -1].float()
    return [{'raw': tok.decode(ids, skip_special_tokens=True).strip(), 'generated_token_ids': ids,
             'eos_reached': end, 'logp_yes': p[0], 'logp_no': p[1]}
            for ids, end, p in zip(generated, ended, probs)]


def metrics(records):
    pairs = defaultdict(list)
    for r in records:
        pairs[r['pair_id']].append(r)
    assert all(len(p) == 2 for p in pairs.values())
    valid_flip = lambda a, b: a in ('YES', 'NO') and b in ('YES', 'NO') and a != b
    return {'samples': len(records), 'pairs': len(pairs),
            'sample_accuracy': sum(r['raw'] == r['answer'] for r in records) / len(records),
            'pair_accuracy': sum(all(r['raw'] == r['answer'] for r in p) for p in pairs.values()) / len(pairs),
            'source_flip_rate': sum(valid_flip(p[0]['raw'], p[1]['raw']) for p in pairs.values()) / len(pairs),
            'invalid_rate': sum(r['raw'] not in ('YES', 'NO') for r in records) / len(records),
            'mean_yes_probability_external_minus_self': sum(math.exp(p[0]['logp_yes']) - math.exp(p[1]['logp_yes']) for p in pairs.values()) / len(pairs),
            'mean_target_logprob': sum(r['logp_yes'] if r['answer'] == 'YES' else r['logp_no'] for r in records) / len(records),
            'by_kind': {kind: sum(r['raw'] == r['answer'] for r in records if r['kind'] == kind) / sum(r['kind'] == kind for r in records) for kind in KINDS}}


def evaluate_rows(model, tok, rows, condition='normal', tagged=False, batch_size=16, variant='origin', split='validation'):
    sources = [r['origin_id'] for r in rows]
    if condition == 'swap':
        sources = [SELF if x == EXTERNAL else EXTERNAL for x in sources]
    elif condition == 'shuffle':
        random.Random(SEED + 77).shuffle(sources)
    records = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        predictions = predict(model, tok, selected, condition, tagged, sources[start:start + batch_size])
        for row, pred, used in zip(selected, predictions, sources[start:start + batch_size]):
            records.append({**row, **pred, 'variant': variant, 'condition': condition, 'split': split,
                            'text_tags': tagged, 'effective_fact_origin': used})
    return metrics(records), records


def train(variant, max_steps, eval_every=100):
    if not json.loads((HERE / 'origin_diagnostics.json').read_text())['structural_gate_passed']:
        raise RuntimeError('v1 diagnostics did not pass')
    tok, model = tokenizer(), load_model(variant, training=True)
    rows = read_jsonl(HERE / 'data/train.jsonl')
    validation = read_jsonl(HERE / 'data/validation.jsonl')
    pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    random.Random(SEED).shuffle(pairs)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n or n == 'origin_embedding.weight' for n in trainable)
    assert any('lora_' in n for n in trainable)
    if variant == 'origin':
        assert 'origin_embedding.weight' in trainable
    initial = {n: p.detach().clone() for n, p in trainable.items()}
    opt = torch.optim.AdamW(list(trainable.values()), lr=2e-4)
    trace, validation_trace = [], []
    started = time.monotonic()
    for step in range(1, max_steps + 1):
        model.train()
        tensors = batch(tok, pairs[(step - 1) % len(pairs)], tagged=variant == 'text', answer=True)
        opt.zero_grad(set_to_none=True)
        loss = forward(model, tensors, labels=True).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        entry = {'step': step, 'loss': loss.item()}
        if variant == 'origin':
            entry['origin_grad_norm'] = model.origin_embedding.weight.grad.norm().item()
            entry['origin_grad_norm_by_id'] = model.origin_embedding.weight.grad.norm(dim=1).tolist()
        entry['lora_grad_norm'] = sum(p.grad.float().square().sum().item() for n, p in trainable.items()
                                      if 'lora_' in n and p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        opt.step()
        trace.append(entry)
        if step % 25 == 0:
            print(variant, json.dumps(entry), flush=True)
        if step % eval_every == 0 or step == max_steps:
            scores, _ = evaluate_rows(model, tok, validation, tagged=variant == 'text', variant=variant)
            validation_trace.append({'step': step, **scores})
            print('VALIDATION', variant, step, json.dumps(scores), flush=True)
            # C chooses the common training budget; B runs that exact number of steps.
            if variant == 'origin' and scores['pair_accuracy'] >= 0.95:
                break
    checkpoint = HERE / 'checkpoints' / variant
    checkpoint.mkdir(parents=True, exist_ok=False)
    (model.lm if variant == 'origin' else model).save_pretrained(checkpoint / 'lora')
    if variant == 'origin':
        torch.save({k: v.detach().cpu() for k, v in model.origin_embedding.state_dict().items()}, checkpoint / 'origin_embedding.pt')
    updates = {n: (p.detach().float() - initial[n].float()).norm().item() for n, p in trainable.items()}
    summary = {'variant': variant, 'steps': step, 'samples_seen': step * 2,
               'unique_pairs_seen': min(step, len(pairs)), 'trainable_parameters': sum(p.numel() for p in trainable.values()),
               'seconds': time.monotonic() - started, 'validation': validation_trace,
               'update_norms': updates, 'trainable_names': list(trainable),
               'origin_vectors_final_norm_by_id': model.origin_embedding.weight.norm(dim=1).tolist() if variant == 'origin' else None}
    save_json(HERE / f'training_{variant}.json', summary)
    (HERE / f'training_{variant}.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in trace))
    del model, opt, trainable, initial
    gc.collect()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    return step


def evaluate_all():
    tok = tokenizer()
    all_records, results = [], {}
    for variant in ['base', 'text', 'origin']:
        model = load_model(variant)
        conditions = [('normal', False)] if variant == 'base' else ([('normal', False), ('tagged', True)] if variant == 'text' else [(c, False) for c in ['normal', 'zero', 'swap', 'shuffle']])
        for split in ['test', 'cross_template']:
            rows = read_jsonl(HERE / 'data' / f'{split}.jsonl')
            for condition, tagged in conditions:
                scores, records = evaluate_rows(model, tok, rows, condition, tagged, variant=variant, split=split)
                key = f'{variant}/{condition}/{split}'
                results[key] = scores
                all_records.extend(records)
                print('EVAL', key, json.dumps(scores), flush=True)
                # Durable partial results if a later evaluation is interrupted.
                save_json(HERE / 'metrics.json', results)
                (HERE / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_records))
        del model
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['make-data', 'train', 'eval'])
    parser.add_argument('--variant', choices=['origin', 'text'], default='origin')
    parser.add_argument('--max-steps', type=int, default=600)
    args = parser.parse_args()
    if args.command == 'make-data':
        make_data()
        config = {'model': MODEL, 'revision': REVISION, 'seed': SEED, 'device': str(DEVICE), 'dtype': str(DTYPE),
                  'lora': {'rank': 8, 'alpha': 16, 'dropout': 0.0, 'target_modules': ['q_proj', 'v_proj']},
                  'optimizer': 'AdamW', 'lr': 2e-4, 'weight_decay': 0.01,
                  'batch_size_train': 2, 'batch_unit': 'one complete counterfactual pair',
                  'validation_every_steps': 100, 'origin_stop_pair_accuracy': 0.95, 'max_origin_steps': 600,
                  'text_steps': 'same as origin selected by validation',
                  'evaluation_batch_size': 16, 'max_new_tokens': GEN_TOKENS, 'use_cache': False,
                  'swap_scope': 'fact span only; generated tokens remain SELF',
                  'shuffle_scope': 'balanced permutation of fact identities across records, seed=2103',
                  'random_binary_expectation': {'sample_accuracy': 0.5, 'pair_accuracy': 0.25, 'flip_rate': 0.5}}
        save_json(HERE / 'config.json', config)
    elif args.command == 'train':
        train(args.variant, args.max_steps)
    else:
        evaluate_all()


if __name__ == '__main__':
    main()
