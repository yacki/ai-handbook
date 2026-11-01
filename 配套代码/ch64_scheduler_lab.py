"""Discrete scheduler and page accounting. Each decode tick costs one unit.
No model/GPU benchmark; prefill and kernel shape costs are intentionally omitted.
"""
import argparse,json,math
from pathlib import Path

def schedule(lengths,capacity,continuous):
    pending=list(enumerate(lengths));active=[];remaining=dict(enumerate(lengths));tick=0;done={};events=[]
    while pending or active:
        if continuous or not active:
            while pending and len(active)<capacity:
                i,_=pending.pop(0);active.append(i)
        tick+=1;running=list(active)
        for i in running:
            remaining[i]-=1
            if remaining[i]==0:done[str(i)]=tick;active.remove(i)
        events.append({'tick':tick,'active_ids':running,'occupied_slots':len(running)})
    return {'ticks':tick,'finish_tick':done,'events':events,'total_output_tokens':sum(lengths),
            'reserved_slot_utilization':sum(lengths)/(tick*capacity)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    lengths=[2,5,3];block_size=16;cache_lengths=[17,33,5];pages=[math.ceil(t/block_size) for t in cache_lengths]
    result={'scope':'discrete toy scheduling and allocation, not measured server speed',
            'static':schedule(lengths,2,False),'continuous':schedule(lengths,2,True),
            'pages':{'lengths':cache_lengths,'block_size':block_size,'page_counts':pages,
                     'used_token_slots':sum(cache_lengths),'allocated_token_slots':sum(pages)*block_size},
            'copy_on_write':{'shared_prefix_blocks':[7,2],'branch_A_tail':9,'branch_B_tail_after_copy':12,
                             'explanation':'two branches share read-only prefix; mutable tail is copied before divergent write'}}
    assert result['static']['ticks']==8 and result['continuous']['ticks']==5
    s=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(s+'\n')
    print(s)
if __name__=='__main__':main()
