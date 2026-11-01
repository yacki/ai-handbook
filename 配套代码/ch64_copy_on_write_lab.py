"""CPU list-based shared KV-page lifecycle, not a tensor cache engine."""
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
 pages={7:list(range(16)),9:[16]};refs={7:2,9:2};branches={'A':[7,9],'B':[7,9]};snapshots=[]
 snapshots.append({'stage':'shared','refs':dict(refs)})
 for name,token in [('A',17),('B',99)]:
  tail=branches[name][-1]
  if refs[tail]>1:
   new=12;pages[new]=list(pages[tail]);refs[new]=1;refs[tail]-=1;branches[name][-1]=new;tail=new
  pages[tail].append(token)
 outputs={name:sum((pages[i] for i in mapping),[]) for name,mapping in branches.items()}
 assert outputs['A'][:-1]==outputs['B'][:-1] and outputs['A'][-1]!=outputs['B'][-1]
 snapshots.append({'stage':'after_divergent_writes','refs':dict(refs),'block_tables':{k:list(v) for k,v in branches.items()}})
 for name in ['A','B']:
  for page in branches.pop(name):
   refs[page]-=1
   if refs[page]==0:del refs[page];del pages[page]
  snapshots.append({'stage':'released_'+name,'refs':dict(refs)})
 assert not pages and not refs
 result={'scope':'CPU page references and copy-on-write only','outputs':outputs,'snapshots':snapshots}
 s=json.dumps(result,ensure_ascii=False,indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
