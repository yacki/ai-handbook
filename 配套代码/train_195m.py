"""Explicit single-CUDA-device 195M entry point, not executed at full scale in this book.
`python train_195m.py --dry-run` only constructs the meta model and reports configuration.
Actual training requires --data, --out, --max-updates, --device cuda:N.
Prepared tensors are loaded in CPU RAM, not a multi-billion-token streaming reader.
"""
import argparse, json, math, random, signal
from dataclasses import asdict
from pathlib import Path
import torch
from book_transformer import BookTransformer, ModelConfig, save_checkpoint, load_checkpoint
from training_project import Cursor, batch_at, filehash, writejson

def run(args):
    cfg=ModelConfig(max_seq_len=args.context,attention_backend='sdpa')
    with torch.device('meta'): probe=BookTransformer(cfg)
    parameters=sum(p.numel() for p in probe.parameters()); assert parameters==195072768
    description={'config':asdict(cfg),'parameters':parameters,'training_executed':False,
                 'data_reader':'prepared dataset.pt loaded in CPU RAM; not streaming',
                 'limitations':['CUDA training not verified in this book','DDP not implemented in this entry point',
                                'SDPA fused-kernel availability and GPU memory must be measured on target hardware']}
    if args.dry_run:
        print(json.dumps(description,ensure_ascii=False,indent=2)); return
    if not args.data or not args.out or not args.device or args.max_updates is None:
        raise ValueError('actual run requires explicit --data --out --device --max-updates')
    if not args.device.startswith('cuda:') or not torch.cuda.is_available(): raise ValueError('explicit available CUDA device required')
    if min(args.max_updates,args.micro_batch,args.accumulation,args.save_every)<=0: raise ValueError('positive run limits required')
    if args.max_train_tokens is not None and args.max_train_tokens<=0: raise ValueError('positive token cap required')
    if not 0<=args.warmup_updates<args.max_updates: raise ValueError('warmup must be shorter than total update budget')
    device=torch.device(args.device); torch.cuda.set_device(device)
    if args.precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('CUDA device does not report BF16 support; select fp32 or another device')
    data_dir=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not args.resume: raise ValueError('output contains a checkpoint; choose a new directory or explicit resume')
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if manifest['vocab_size']!=32768 or manifest['sequence_length']!=args.context: raise ValueError('formal vocabulary/context and prepared data disagree')
    for filename,key in [('dataset.pt','dataset_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if filehash(data_dir/filename)!=manifest[key]: raise ValueError('prepared artifact hash mismatch')
    data=torch.load(data_dir/'dataset.pt',map_location='cpu',weights_only=True)['train']
    if len(data['labels'])==0: raise ValueError('empty training windows')
    torch.manual_seed(args.seed); random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model=BookTransformer(cfg).to(device); model.gradient_checkpointing=args.checkpoint_blocks; model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    def multiplier(step):
        if step<args.warmup_updates: return (step+1)/max(1,args.warmup_updates)
        progress=min(1.0,(step-args.warmup_updates)/max(1,args.max_updates-args.warmup_updates))
        return 0.1+0.9*0.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    plan={key:getattr(args,key) for key in ['context','max_updates','max_train_tokens','micro_batch','accumulation','precision',
          'checkpoint_blocks','lr','weight_decay','warmup_updates','seed']}
    metadata={'manifest_sha256':filehash(data_dir/'manifest.json'),'training_plan':plan}
    cursor=Cursor(len(data['labels']),seed=args.seed); step=0; tokens=0
    if args.resume:
        state=load_checkpoint(args.resume,model,optimizer,scheduler=scheduler,expected_metadata=metadata)
        cursor=Cursor(len(data['labels']),state=state['sampler_state']); step=state['step']; tokens=state['tokens']
    stop={'requested':False}
    def request_stop(signum,frame): stop['requested']=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    def persist():
        save_checkpoint(out/'checkpoint.pt',model,optimizer,step,tokens,cursor.state(),scheduler=scheduler,metadata=metadata)
    empty_attempts=0; reason='update_budget'; optimizer.zero_grad(set_to_none=True)
    with (out/'metrics.jsonl').open('a') as log:
        while step<args.max_updates:
            if stop['requested']: reason='signal_at_update_boundary'; break
            # Retain the prefetch cursor if the token cap prevents this update.
            cursor_before=cursor.state()
            cpu_batches=[batch_at(data,cursor.next(args.micro_batch)) for _ in range(args.accumulation)]
            count=sum(int(b['labels'].ne(-100).sum()) for b in cpu_batches)
            if args.max_train_tokens is not None and tokens+count>args.max_train_tokens:
                cursor=Cursor(len(data['labels']),state=cursor_before); reason='token_cap_before_next_update'; break
            if count==0:
                empty_attempts+=1
                if empty_attempts>100: raise ValueError('100 empty update windows; inspect target masks')
                continue
            empty_attempts=0; numerator=0.0
            for cpu_batch in cpu_batches:
                batch={key:value.to(device) for key,value in cpu_batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                    result=model(**batch); normalized=result['loss_sum']/count
                if not torch.isfinite(normalized): raise FloatingPointError('nonfinite loss; restore previous checkpoint after diagnosis')
                normalized.backward(); numerator+=float(result['loss_sum'].detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step+=1; tokens+=count
            record={'step':step,'effective_tokens':tokens,'update_tokens':count,'loss':numerator/count,
                    'gradient_norm_before_clip':float(norm),'lr_next_update':optimizer.param_groups[0]['lr']}
            log.write(json.dumps(record)+'\n'); log.flush()
            if step%args.save_every==0: persist()
    persist()
    writejson(out/'run_status.json',{**description,'training_executed':True,'completed_updates':step,
              'effective_training_tokens':tokens,'stop_reason':reason,'metadata':metadata})
    print(json.dumps({'completed_updates':step,'tokens':tokens,'stop_reason':reason},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); p.add_argument('--data'); p.add_argument('--out'); p.add_argument('--device')
    p.add_argument('--context',type=int,choices=[1024,2048],default=1024); p.add_argument('--max-updates',type=int); p.add_argument('--max-train-tokens',type=int)
    p.add_argument('--micro-batch',type=int,default=1); p.add_argument('--accumulation',type=int,default=1); p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--checkpoint-blocks',action='store_true'); p.add_argument('--resume'); p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=0.1); p.add_argument('--warmup-updates',type=int,default=0); p.add_argument('--seed',type=int,default=20261101)
    run(p.parse_args())
