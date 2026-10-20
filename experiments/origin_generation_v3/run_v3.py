"""Single-turn source-grounded generation pilot; previous experiments are read-only."""
import argparse, gc, hashlib, json, math, os, random, re, sys, time
# Offline mode is controlled by the caller in the release copy.
from collections import Counter, defaultdict
from pathlib import Path
import torch
from peft import PeftModel
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/origin_counterfactual_v2'))
import run_experiment as v2
SYSTEM,USER,EXTERNAL,SELF=range(4)
SEED=31019
MAX_NEW=96
KEYS={'database':['PostgreSQL','MySQL','SQLite','MariaDB'],
      'pool limit':['8','16','24','32','48','64'],
      'cache policy':['fifo','lru','lfu','disabled','write-through','write-back'],
      'log path':None,'release date':None,'owner':None}
QUESTIONS=[
 'Describe the project configuration and distinguish confirmed information from unverified claims.',
 'Summarize what is established about this project and what remains uncertain.',
 'Explain the known configuration. Keep any unconfirmed claims clearly qualified.',
 'Give an account of this project, separating documented facts from tentative claims.',
 'What can be stated confidently about this project, and what should still be treated as unconfirmed?']
SPECULATE=[
 'Include the suggested possibilities, but identify them as unconfirmed.',
 'Also retain the proposed inferences and explain their uncertainty.',
 'Discuss the supplied possibilities without presenting them as established facts.',
 'Mention the candidate explanations while keeping their tentative status clear.',
 'Please preserve the proposed possibilities as possibilities, rather than established information.']
PROMPT='Use the reference as evidence. Distinguish confirmed statements, uncertain statements in the reference, and unverified inferences. Answer in two short sentences, covering both supplied claims. Do not add new claims.\n'


def save(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def read(name): return [json.loads(x) for x in (HERE/name).read_text().splitlines()]
def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def expected(atom,source):
    if source==SELF:return 'inference'
    return 'uncertain_reference' if atom['uncertain'] else 'confirmed'

def target(row,swap=False):
    sentences=[]
    for a,src in zip(row['atoms'],row['sources']):
        if swap:src=SELF if src==EXTERNAL else EXTERNAL
        claim=f"the {a['key']} is {a['value']}"
        status=expected(a,src)
        if status=='confirmed':s=f'The reference confirms that {claim}.'
        elif status=='uncertain_reference':s=f'The reference suggests that {claim}, but this is not confirmed.'
        else:s=f'It is possible that {claim}, but this is not confirmed by the reference.'
        sentences.append(s)
    return ' '.join(sentences)

def statement(a,style):
    claim=f"the {a['key']} is {a['value']}"
    if style==0:return ('It is possible that '+claim if a['uncertain'] else claim.capitalize())+'.'
    if style==1:return (f"The {a['key']} might be {a['value']}." if a['uncertain'] else f"For this project, {claim}.")
    if style==2:return (f"A tentative value for {a['key']} is {a['value']}." if a['uncertain'] else f"The configured {a['key']} is {a['value']}.")
    return (f"Possibly, {claim}." if a['uncertain'] else f"This project's {a['key']} has the value {a['value']}.")

def encode(tok,row,tagged=False,answer=False,swap=False):
    parts=[(SYSTEM,PROMPT)]
    for a,src in zip(row['atoms'],row['sources']):
        if swap:src=SELF if src==EXTERNAL else EXTERNAL
        tag=('[EXTERNAL] ' if src==EXTERNAL else '[SELF] ') if tagged else ''
        parts.append((src,tag+a['text']+'\n'))
    parts += [(USER,row['question']+'\n'),(SELF,'Answer:\n')]
    ids=[];orig=[]
    for src,text in parts:
        p=tok.encode(text,add_special_tokens=False);ids+=p;orig += [src]*len(p)
    n=len(ids)
    if answer:
        p=tok.encode(target(row),add_special_tokens=False)+[tok.eos_token_id]
        ids+=p;orig += [SELF]*len(p)
    assert len(ids)<=384
    return ids,orig,[-100]*n+ids[n:]

def batch(tok,rows,tagged=False,answer=False,swap=False):
    encoded=[encode(tok,r,tagged,answer,swap) for r in rows]
    width=max(len(x[0]) for x in encoded)
    fields=[[],[],[],[]]
    for ids,orig,labels in encoded:
        pad=width-len(ids)
        for dst,val in zip(fields,[[tok.eos_token_id]*pad+ids,[SYSTEM]*pad+orig,[0]*pad+[1]*len(ids),[-100]*pad+labels]):dst.append(val)
    return dict(zip(['ids','origins','mask','labels'],[torch.tensor(x,device=v2.DEVICE) for x in fields]))

def forward(model,b,zero=False,train=False,past=None,cache=False):
    ids,orig=b['ids'],b['origins']
    positions=(b['mask'].cumsum(-1)-1).clamp_min(0)
    if past is not None:ids=ids[:,-1:];orig=orig[:,-1:];positions=positions[:,-1:]
    lm=model.lm if isinstance(model,v2.OriginModel) else model
    kw={'attention_mask':b['mask'],'position_ids':positions,'use_cache':cache}
    if train:kw['labels']=b['labels']
    if past is not None:kw['past_key_values']=past
    if isinstance(model,v2.OriginModel):
        embeds=lm.get_input_embeddings()(ids)
        if not zero:embeds=embeds+model.origin_embedding(orig).to(embeds.dtype)
        kw['inputs_embeds']=embeds
    else:kw['input_ids']=ids
    return lm(**kw)

@torch.no_grad()
def generate(model,tok,rows,tagged=False,zero=False,swap=False,cache=True,max_new=MAX_NEW):
    model.eval();b=batch(tok,rows,tagged,swap=swap);past=None
    tokens=[[] for _ in rows];ended=[False]*len(rows)
    for _ in range(max_new):
        out=forward(model,b,zero=zero,past=past,cache=cache)
        if cache:past=out.past_key_values
        ids=out.logits[:,-1].argmax(-1)
        for i,t in enumerate(ids.tolist()):
            if not ended[i]:
                if t==tok.eos_token_id:ended[i]=True
                else:tokens[i].append(t)
        if all(ended):break
        b['ids']=torch.cat([b['ids'],ids[:,None]],1)
        b['origins']=torch.cat([b['origins'],torch.full_like(ids[:,None],SELF)],1)
        b['mask']=torch.cat([b['mask'],torch.ones_like(ids[:,None])],1)
    return [{'raw':tok.decode(t,skip_special_tokens=True).strip(),'token_ids':t,'eos':e} for t,e in zip(tokens,ended)]

def make_data():
    tok=v2.tokenizer();rng=random.Random(SEED);seen=set();validation={}
    (HERE/'data').mkdir(exist_ok=True)
    for split,count in [('train',320),('validation',24),('transfer',12),('test',48),('cross_template',48)]:
        rows=[]
        for i in range(count):
            case=['mixed','conflict','uncertain'][i%3]
            key=list(KEYS)[(i//3)%6]
            key2=key if case=='conflict' else list(KEYS)[(list(KEYS).index(key)+1)%6]
            entity=f'project_{rng.getrandbits(48):012x}'
            assert entity not in seen;seen.add(entity)
            def value(k):
                if KEYS[k]:return rng.choice(KEYS[k])
                if k=='log path':return f'/srv/{rng.getrandbits(40):010x}/events.log'
                if k=='owner':return f'team_{rng.getrandbits(32):08x}'
                return f'{rng.randrange(2025,2040)}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}'
            va,vb=value(key),value(key2)
            while va==vb:vb=value(key2)
            uncertain=[False,False] if case!='uncertain' else [False,True]
            if case=='mixed' and i%2==0:uncertain=[True,False]
            style=3 if split=='cross_template' else rng.randrange(3)
            atoms=[{'key':k,'value':v,'uncertain':u} for k,v,u in [(key,va,uncertain[0]),(key2,vb,uncertain[1])]]
            # Include the entity in the task, not in the answer: each fact-content tuple is unique.
            for a in atoms:a['text']=statement(a,style)
            if rng.randrange(2):atoms.reverse()
            qi=4 if split=='cross_template' else (3 if split=='validation' else rng.randrange(3))
            speculate=(i%2==0)
            question=f'For {entity}: '+QUESTIONS[qi]+(' '+SPECULATE[qi] if speculate else '')
            for source in [EXTERNAL,SELF]:
                rows.append({'pair_id':f'{split}-{i:04d}','entity':entity,'case':case,'atoms':atoms,
                             'sources':[source,SELF if source==EXTERNAL else EXTERNAL],
                             'question':question,'speculate':speculate,'fact_style':style,'question_style':qi})
        for a,b in zip(rows[::2],rows[1::2]):
            ia,oa,_=encode(tok,a);ib,ob,_=encode(tok,b)
            assert ia==ib and oa!=ob and target(a)!=target(b)
            assert encode(tok,a,swap=True)[1]==ob
        (HERE/f'data/{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        validation[split]={'pairs':count,'records':len(rows),'sha256':hashfile(HERE/f'data/{split}.jsonl')}
    # Generic database/config values can repeat; entity-qualified facts cannot overlap.
    save('data-validation.json',{'splits':validation,'unique_entities':len(seen),'pair_token_identity':True,
                                'swap_matches_partner':True,'value_domains_shared':True})
    save('config.json',{'model':v2.MODEL,'revision':v2.REVISION,'seed':SEED,'adapter_init_seed':v2.SEED,'offline_cached_revision':True,'steps':200,'batch_size':2,
        'lr':2e-4,'lora_rank':8,'lora_alpha':16,'target_modules':['q_proj','v_proj'],
        'max_new_tokens':MAX_NEW,'greedy':True,'cache':True,'checkpoint_selection':'fixed final step 200; validation does not select test results',
        'base_condition':'untuned pretrained model WITH the same textual source tags as B',
        'scope':'single turn, two supplied claims; inference retention refers to supplied candidate inferences, not novel deductive reasoning'})
    print('DATA',json.dumps(validation),flush=True)

def score(row,raw,swap=False):
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',raw) if s.strip()]
    findings=[];matched=set()
    for i,a in enumerate(row['atoms']):
        hits=[]
        for j,s in enumerate(sentences):
            if re.search(r'(?<![\w/.-])'+re.escape(a['value'])+r'(?![\w/-]|\.[\w/-])',s,re.I):hits.append((j,s));matched.add(j)
        status='omitted'
        if hits:
            statuses=[]
            for j,s in hits:
                low=s.lower()
                if '?' in s or '[SELF]' in s or '[EXTERNAL]' in s:
                    statuses.append('unresolved');continue
                if any(re.search(r'(?<![\w/.-])'+re.escape(b['value'])+r'(?![\w/-]|\.[\w/-])',s,re.I) for n,b in enumerate(row['atoms']) if n!=i):statuses.append('unresolved');continue
                # Limit automatic scoring to recognizable claim-bearing sentences.
                if a['key'].lower() not in low:statuses.append('unresolved');continue
                uncertain=bool(re.search(r'\b(possible|possibly|might|may|could|tentative|uncertain|unverified|inference|not confirmed)\b',low))
                if 'reference suggests' in low:statuses.append('uncertain_reference')
                elif uncertain:statuses.append('inference')
                elif re.search(r'\b(not|never|no)\b',low):statuses.append('unresolved')
                else:statuses.append('confirmed')
            status=statuses[0] if len(set(statuses))==1 else 'unresolved'
        src=row['sources'][i]
        if swap:src=SELF if src==EXTERNAL else EXTERNAL
        findings.append({'key':a['key'],'value':a['value'],'source':src,'expected':expected(a,src),'observed':status})
    extra=[s for j,s in enumerate(sentences) if j not in matched]
    return {'atoms':findings,'unparsed_sentences':extra,
            'fully_scorable':not extra and all(x['observed'] not in ['omitted','unresolved'] for x in findings),
            'exact_target':raw==target(row,swap)}

def aggregate(records):
    totals=Counter();bycase=defaultdict(Counter)
    for r in records:
        s=r['score'];totals['records']+=1;totals['fully_scorable']+=s['fully_scorable'];totals['exact_target']+=s['exact_target']
        totals['unparsed_records']+=bool(s['unparsed_sentences'])
        for a in s['atoms']:
            e,o=a['expected'],a['observed'];totals['atoms']+=1
            totals['omitted']+=o=='omitted';totals['unresolved']+=o=='unresolved'
            if e=='confirmed':totals['external_total']+=1;totals['external_retained']+=o=='confirmed'
            else:
                totals['unsupported_total']+=1;totals['asserted_without_support']+=o=='confirmed'
                totals['unsupported_unresolved']+=o=='unresolved'
            if a['source']==SELF:
                totals['self_total']+=1;totals['self_promoted']+=o in ['confirmed','uncertain_reference']
                if r['speculate']:totals['requested_inference_total']+=1;totals['requested_inference_retained']+=o=='inference'
            if e=='uncertain_reference':totals['uncertain_external_total']+=1;totals['uncertain_external_retained']+=o=='uncertain_reference'
        bycase[r['case']]['records']+=1
        bycase[r['case']]['exact_target']+=s['exact_target']
    ratio=lambda a,b:totals[a]/totals[b] if totals[b] else None
    return {'counts':dict(totals),'fully_scorable_rate':ratio('fully_scorable','records'),
        'exact_target_rate':ratio('exact_target','records'),
        'unsupported_assertion_rate_lower_bound':ratio('asserted_without_support','unsupported_total'),
        'self_misattribution_rate_lower_bound':ratio('self_promoted','self_total'),
        'external_fact_retention':ratio('external_retained','external_total'),
        'requested_supplied_inference_retention':ratio('requested_inference_retained','requested_inference_total'),
        'uncertain_external_retention':ratio('uncertain_external_retained','uncertain_external_total'),
        'by_case_exact':{k:dict(v) for k,v in bycase.items()}}

def evaluate(model,tok,rows,tagged=False,zero=False,swap=False,name='evaluation'):
    records=[]
    for start in range(0,len(rows),8):
        group=rows[start:start+8]
        for r,p in zip(group,generate(model,tok,group,tagged,zero,swap)):
            records.append({**r,**p,'condition':name,'score':score(r,p['raw'],swap)})
    metrics=aggregate(records)
    print(name,json.dumps(metrics),flush=True)
    return metrics,records

def load(variant,checkpoint='v3',training=False):
    if training:return v2.load_model(variant,training=True)
    if variant=='base' or checkpoint=='v2':return v2.load_model(variant)
    lm=v2.load_model('base')
    lm=PeftModel.from_pretrained(lm,HERE/'checkpoints'/variant/'lora')
    model=v2.OriginModel(lm) if variant=='origin' else lm
    if variant=='origin':model.origin_embedding.load_state_dict(torch.load(HERE/'checkpoints/origin/origin_embedding.pt',map_location='cpu',weights_only=True))
    return model.to(v2.DEVICE)

def release(model):
    del model;gc.collect()
    if v2.DEVICE.type=='mps':torch.mps.empty_cache()

def transfer():
    tok=v2.tokenizer();rows=read('data/transfer.jsonl');summary={};records=[]
    for variant in ['text','origin']:
        model=load(variant,'v2')
        # Validate standard KV caching against full-prefix recomputation on same input.
        a=generate(model,tok,rows[:2],tagged=variant=='text',max_new=16)
        b=generate(model,tok,rows[:2],tagged=variant=='text',cache=False,max_new=16)
        assert [r['token_ids'] for r in a]==[r['token_ids'] for r in b], 'KV cache parity failed'
        name='v2_'+variant
        m,p=evaluate(model,tok,rows,tagged=variant=='text',name=name);summary[name]=m;records+=p
        if variant=='origin':
            m,p=evaluate(model,tok,rows,zero=True,name='v2_origin_zero');summary['v2_origin_zero']=m;records+=p
        del model;gc.collect()
        if v2.DEVICE.type=='mps':torch.mps.empty_cache()
    save('transfer-results.json',{'metrics':summary,'cache_parity_passed':True,'training_performed':False})
    (HERE/'transfer-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

def train(variant):
    torch.manual_seed(SEED);tok=v2.tokenizer();model=load(variant,training=True)
    torch.manual_seed(SEED)
    rows=read('data/train.jsonl');pairs=[rows[i:i+2] for i in range(0,len(rows),2)]
    random.Random(SEED).shuffle(pairs)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n or n=='origin_embedding.weight' for n in params)
    initial={n:p.detach().clone() for n,p in params.items()}
    opt=torch.optim.AdamW(list(params.values()),lr=2e-4)
    traces=[];vals=[];start=time.monotonic();prompt_tokens=answer_tokens=0
    for step in range(1,201):
        model.train();group=pairs[step-1];b=batch(tok,group,tagged=variant=='text',answer=True)
        prompt_tokens+=int(((b['labels']==-100)&(b['mask']==1)).sum());answer_tokens+=int((b['labels']!=-100).sum())
        opt.zero_grad(set_to_none=True);loss=forward(model,b,train=True).loss
        assert torch.isfinite(loss)
        loss.backward();entry={'step':step,'loss':loss.item()}
        if variant=='origin':entry['origin_grad_norm']=model.origin_embedding.weight.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.0,error_if_nonfinite=True);opt.step();traces.append(entry)
        if step%25==0:print('TRAIN',variant,entry,flush=True)
        if step in [100,200]:
            m,p=evaluate(model,tok,read('data/validation.jsonl'),tagged=variant=='text',name=f'{variant}_validation_{step}')
            vals.append({'step':step,**m})
            (HERE/f'validation-{variant}-{step}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in p))
    ck=HERE/'checkpoints'/variant;ck.mkdir(parents=True,exist_ok=False)
    (model.lm if variant=='origin' else model).save_pretrained(ck/'lora')
    if variant=='origin':torch.save({k:v.detach().cpu() for k,v in model.origin_embedding.state_dict().items()},ck/'origin_embedding.pt')
    save(f'training-{variant}.json',{'steps':200,'pairs_seen':200,'samples_seen':400,'prompt_tokens':prompt_tokens,'answer_tokens':answer_tokens,
        'seconds':time.monotonic()-start,'validation':vals,'trainable_parameters':sum(p.numel() for p in params.values()),
        'updates':{n:(p.detach().float()-initial[n].float()).norm().item() for n,p in params.items()}})
    (HERE/f'training-{variant}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))

def evaluate_all():
    tok=v2.tokenizer();metrics={};records=[];cache_checks={}
    for variant in ['base','text','origin']:
        model=load(variant)
        probe=read('data/test.jsonl')[:2]
        cached=generate(model,tok,probe,tagged=variant!='origin',max_new=32)
        uncached=generate(model,tok,probe,tagged=variant!='origin',cache=False,max_new=32)
        cache_checks[variant]=[r['token_ids'] for r in cached]==[r['token_ids'] for r in uncached]
        assert cache_checks[variant], 'KV cache parity failed'
        for split in ['test','cross_template']:
            for condition in (['normal','zero','swap'] if variant=='origin' else ['normal']):
                name=f'{variant}/{condition}/{split}'
                m,p=evaluate(model,tok,read(f'data/{split}.jsonl'),tagged=variant!='origin',zero=condition=='zero',swap=condition=='swap',name=name)
                metrics[name]=m;records+=p
                save('metrics.json',metrics)
                (HERE/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        del model;gc.collect()
        if v2.DEVICE.type=='mps':torch.mps.empty_cache()
    save('cache-checks.json',cache_checks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['make-data','transfer','train','eval']);p.add_argument('--variant',choices=['text','origin'],default='origin');a=p.parse_args()
    if a.command=='make-data':make_data()
    elif a.command=='transfer':transfer()
    elif a.command=='train':train(a.variant)
    else:evaluate_all()
