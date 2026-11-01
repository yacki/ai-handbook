"""Ch33-43 CPU teaching project: own byte-level BPE -> packed tensors -> training -> resume.
Requires torch and tokenizers. No downloads. `demo --out DIR` is a bounded CPU experiment.
`prepare` accepts real JSONL and a 32768-word vocabulary, but refuses underfilled vocabularies.
The train command deliberately supports tiny CPU only; formal 195M runs need a budgeted trainer.
"""
import argparse, hashlib, json, math, random
from dataclasses import asdict
from pathlib import Path
import torch
import tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from book_transformer import BookTransformer, ModelConfig, tiny_config, save_checkpoint, load_checkpoint

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']
def digest(data): return hashlib.sha256(data).hexdigest()
def filehash(path): return digest(Path(path).read_bytes())
def writejson(path, obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def read_docs(path):
    docs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not docs: raise ValueError('empty split')
    seen = set()
    for doc in docs:
        for field in ('id','text','source','license'):
            if field not in doc: raise ValueError('missing '+field)
        if doc['id'] in seen: raise ValueError('duplicate document id')
        seen.add(doc['id'])
        if not doc['text'].strip(): raise ValueError('empty document')
        if any(mark in doc['text'] for mark in SPECIAL): raise ValueError('reserved token literal in source text')
    return docs

def make_windows(docs, tokenizer, length):
    # Independent documents, EOS plus BOS; cross-document labels are ignored.
    stream, owners = [], []
    for i,doc in enumerate(docs):
        ids = [tokenizer.token_to_id('<bos>')] + tokenizer.encode(doc['text'],add_special_tokens=False).ids + [tokenizer.token_to_id('<eos>')]
        stream.extend(ids); owners.extend([i]*len(ids))
    rows, labels, groups = [], [], []
    for start in range(0,len(stream)-1,length):
        seq = stream[start:start+length+1]; docids = owners[start:start+length+1]
        x, y, d = seq[:-1], seq[1:], docids[:-1]
        y = [target if docids[j]==docids[j+1] else -100 for j,target in enumerate(y)]
        if not any(t != -100 for t in y): continue
        pad = length-len(x)
        rows.append(x+[tokenizer.token_to_id('<pad>')]*pad)
        labels.append(y+[-100]*pad); groups.append(d+[-1]*pad)
    if not rows: raise ValueError('no training transitions')
    return {'input_ids':torch.tensor(rows,dtype=torch.long), 'labels':torch.tensor(labels,dtype=torch.long),
            'document_ids':torch.tensor(groups,dtype=torch.long)}

def prepare(train_path, valid_path, out, vocab_size=32768, length=1024):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    train, valid = read_docs(train_path), read_docs(valid_path)
    a = {digest(d['text'].encode()) for d in train}; b = {digest(d['text'].encode()) for d in valid}
    if a & b: raise ValueError('exact text leakage across splits')
    if {d['id'] for d in train} & {d['id'] for d in valid}: raise ValueError('document id leakage')
    if vocab_size < 260: raise ValueError('byte alphabet plus special tokens need at least 260 entries')
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,special_tokens=SPECIAL,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),show_progress=False)
    tok.train_from_iterator((d['text'] for d in train),trainer=trainer,length=len(train))
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f'corpus produced {tok.get_vocab_size()} entries, requested {vocab_size}; expand training corpus or explicitly choose a smaller experiment')
    tok.save(str(out/'tokenizer.json'))
    probe='退款 amount=12.50\nemoji🙂 and café'
    assert tok.decode(tok.encode(probe,add_special_tokens=False).ids)==probe
    tensors = {name:make_windows(docs,tok,length) for name,docs in [('train',train),('valid',valid)]}
    torch.save(tensors,out/'dataset.pt')
    manifest = {'schema_version':1,'vocab_size':vocab_size,'sequence_length':length,
                'tokenizers_version':tokenizers.__version__,'normalization':'none','packing':'same-document causal; cross-document targets ignored',
                'tokenizer_sha256':filehash(out/'tokenizer.json'),'dataset_sha256':filehash(out/'dataset.pt'),
                'source_files':{'train':filehash(train_path),'valid':filehash(valid_path)},
                'documents':{name:[{'id':d['id'],'source':d['source'],'license':d['license'],'text_sha256':digest(d['text'].encode())} for d in docs]
                             for name,docs in [('train',train),('valid',valid)]},
                'windows':{name:int(data['labels'].shape[0]) for name,data in tensors.items()},
                'effective_tokens':{name:int(data['labels'].ne(-100).sum()) for name,data in tensors.items()}}
    writejson(out/'manifest.json',manifest); return manifest

class Cursor:
    def __init__(self,size,seed=17,state=None):
        self.size=size; self.gen=torch.Generator().manual_seed(seed); self.epoch=0; self.position=0
        self.order=torch.randperm(size,generator=self.gen)
        if state is not None:
            self.order=torch.tensor(state['order']); self.position=state['position']; self.epoch=state['epoch']; self.gen.set_state(state['generator'])
    def next(self,batch_size):
        if self.position==self.size:
            self.order=torch.randperm(self.size,generator=self.gen); self.position=0; self.epoch+=1
        result=self.order[self.position:min(self.position+batch_size,self.size)]; self.position+=len(result); return result
    def state(self): return {'order':self.order.tolist(),'position':self.position,'epoch':self.epoch,'generator':self.gen.get_state()}

def batch_at(data,indices): return {name:tensor[indices] for name,tensor in data.items()}

def accumulated_update(model,optimizer,microbatches,amp=False):
    # Sum losses / ONE effective-token denominator, with parameters fixed until step.
    count=sum(int((b['labels'].ne(-100) & (b['loss_mask'].bool() if 'loss_mask' in b else torch.ones_like(b['labels'],dtype=torch.bool))).sum()) for b in microbatches)
    optimizer.zero_grad(set_to_none=True)
    if count==0: return {'updated':False,'tokens':0,'loss':None}
    total=0.0
    for batch in microbatches:
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
            result=model(**batch); loss=result['loss_sum']/count
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward(); total+=float(result['loss_sum'].detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {'updated':True,'tokens':count,'loss':total/count,'grad_norm':float(norm)}

@torch.no_grad()
def evaluate(model,data,batch_size=4):
    model.eval(); numerator=0.0; denominator=0
    for start in range(0,len(data['labels']),batch_size):
        b={key:value[start:start+batch_size] for key,value in data.items()}; r=model(**b)
        numerator+=float(r['loss_sum']); denominator+=int(r['token_count'])
    model.train()
    if denominator==0: raise ValueError('validation has no effective targets')
    return numerator/denominator

def train_project(data_dir,out,updates=30,stop_after=None,resume=None,amp=False):
    torch.set_num_threads(1); torch.manual_seed(910); random.seed(910)
    data_dir=Path(data_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if filehash(data_dir/'dataset.pt')!=manifest['dataset_sha256'] or filehash(data_dir/'tokenizer.json')!=manifest['tokenizer_sha256']:
        raise ValueError('prepared data integrity mismatch')
    if manifest['sequence_length']>64: raise ValueError('CPU tiny trainer refuses windows longer than 64')
    tensors=torch.load(data_dir/'dataset.pt',map_location='cpu',weights_only=True)
    cfg=tiny_config(); cfg.vocab_size=manifest['vocab_size']; cfg.max_seq_len=manifest['sequence_length']; cfg.attention_backend='sdpa'
    model=BookTransformer(cfg); opt=torch.optim.AdamW(model.parameters(),lr=0.006,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=updates,eta_min=0.0006)
    metadata={'manifest_sha256':filehash(data_dir/'manifest.json'),'updates':updates,'batch_size':2,'accumulation':2,'amp':amp}
    cursor=Cursor(len(tensors['train']['labels'])); step=0; tokens=0
    if resume:
        state=load_checkpoint(resume,model,opt,scheduler=scheduler,expected_metadata=metadata)
        cursor=Cursor(len(tensors['train']['labels']),state=state['sampler_state']); step=state['step']; tokens=state['tokens']
    initial=evaluate(model,tensors['train']); losses=[]; limit=stop_after if stop_after is not None else updates
    attempts=0
    while step<limit:
        micro=[batch_at(tensors['train'],cursor.next(2)) for _ in range(2)]
        result=accumulated_update(model,opt,micro,amp=amp); attempts+=1
        if attempts>updates*20: raise RuntimeError('too many empty updates')
        if not result['updated']: continue
        scheduler.step(); step+=1; tokens+=result['tokens']; losses.append(result['loss'])
    checkpoint_path=out/'checkpoint.pt'
    save_checkpoint(checkpoint_path,model,opt,step,tokens,cursor.state(),scheduler=scheduler,metadata=metadata)
    report={'architecture':'tiny','parameters':sum(p.numel() for p in model.parameters()),'device':'cpu','amp_bfloat16':amp,
            'step':step,'training_tokens':tokens,'train_loss_at_entry':initial,'train_loss':evaluate(model,tensors['train']),
            'valid_loss':evaluate(model,tensors['valid']),'config':asdict(cfg),'checkpoint':str(checkpoint_path)}
    writejson(out/'report.json',report); return model,opt,report

def demo(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Synthetic authored text; held-out identifiers differ, domain and templates intentionally overlap.
    for split,count,start in [('train',96,0),('valid',16,1000)]:
        docs=[]
        for i in range(start,start+count):
            color=['红色','蓝色','绿色'][i%3]; owner=['小林','小周','小陈'][i%3]
            docs.append({'id':f'{split}-{i}','text':f'记录{i}：{color}订单由{owner}负责。状态是已确认。\nOrder {i}: owner confirmed; amount={i%7+10}.00. Please retain the reference number.',
                         'source':'authored-synthetic-demo','license':'authored for this tutorial'})
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    manifest=prepare(out/'train.jsonl',out/'valid.jsonl',out/'prepared',vocab_size=320,length=32)
    full,_,a=train_project(out/'prepared',out/'continuous',updates=30)
    train_project(out/'prepared',out/'interrupted',updates=30,stop_after=15)
    resumed,_,b=train_project(out/'prepared',out/'resumed',updates=30,resume=out/'interrupted'/'checkpoint.pt')
    difference=max((p-q).abs().max().item() for p,q in zip(full.parameters(),resumed.parameters()))
    assert difference==0.0 and a['training_tokens']==b['training_tokens']
    # Empty batch must not advance AdamW or decay weights.
    cfg=tiny_config(); m=BookTransformer(cfg); o=torch.optim.AdamW(m.parameters(),lr=0.01,weight_decay=0.1)
    ids=torch.ones(1,4,dtype=torch.long); empty={'input_ids':ids,'labels':torch.full_like(ids,-100)}
    accumulated_update(m,o,[{'input_ids':ids,'labels':torch.full_like(ids,2)}])
    steps_before=[int(state['step']) for state in o.state.values()]
    before=[p.detach().clone() for p in m.parameters()]; empty_result=accumulated_update(m,o,[empty])
    zero_difference=max((p-q).abs().max().item() for p,q in zip(m.parameters(),before))
    unchanged_steps=steps_before==[int(state['step']) for state in o.state.values()]
    assert zero_difference==0 and unchanged_steps
    # Compare full batch and differently sized microbatches with differing valid counts.
    x=torch.tensor([[1,2,3,4],[2,3,4,1],[3,4,1,2]]); y=torch.tensor([[2,3,4,1],[3,4,-100,-100],[4,1,2,3]])
    p=BookTransformer(cfg); q=BookTransformer(cfg); q.load_state_dict(p.state_dict())
    p(x,y)['loss'].backward(); count=int(y.ne(-100).sum())
    for lo,hi in [(0,1),(1,3)]: (q(x[lo:hi],y[lo:hi])['loss_sum']/count).backward()
    grad_diff=max((u.grad-v.grad).abs().max().item() for u,v in zip(p.parameters(),q.parameters())); assert grad_diff<1e-5
    # Actual CPU bf16 forward/backward only; no GPU or speed claim.
    bf=BookTransformer(cfg); bo=torch.optim.AdamW(bf.parameters(),lr=0.001)
    bf_result=accumulated_update(bf,bo,[{'input_ids':x,'labels':y}],amp=True)
    result={'torch_version':torch.__version__,'tokenizers_version':tokenizers.__version__,
            'vocab_size':manifest['vocab_size'],'windows':manifest['windows'],'effective_tokens':manifest['effective_tokens'],
            'continuous':a,'resumed':b,'resume_max_parameter_diff':difference,'zero_token_update':empty_result,
            'zero_token_parameter_diff':zero_difference,'zero_token_optimizer_steps_unchanged':unchanged_steps,'accumulation_max_gradient_diff':grad_diff,
            'cpu_bfloat16_update':bf_result,'formal_195m_training_executed':False,'ddp_executed':False}
    writejson(out/'project_verification.json',result); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('demo'); p.add_argument('--out',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--train-jsonl',required=True); p.add_argument('--valid-jsonl',required=True); p.add_argument('--out',required=True); p.add_argument('--vocab-size',type=int,default=32768); p.add_argument('--length',type=int,default=1024)
    p=sub.add_parser('train'); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--updates',type=int,default=30); p.add_argument('--resume'); p.add_argument('--amp-bf16',action='store_true')
    p=sub.add_parser('meta'); p.add_argument('--context',type=int,default=1024)
    args=parser.parse_args()
    if args.command=='demo': demo(args.out)
    elif args.command=='prepare': prepare(args.train_jsonl,args.valid_jsonl,args.out,args.vocab_size,args.length)
    elif args.command=='train': train_project(args.data,args.out,args.updates,resume=args.resume,amp=args.amp_bf16)
    else:
        cfg=ModelConfig(max_seq_len=args.context,attention_backend='sdpa')
        with torch.device('meta'): model=BookTransformer(cfg)
        print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'config':asdict(cfg),'training_executed':False},indent=2))
