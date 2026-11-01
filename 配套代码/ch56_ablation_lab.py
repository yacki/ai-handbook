"""Paired statistics on explicit synthetic observations; no training."""
import json,statistics,math,argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
 base=[3.10,3.00,3.20];candidate=[3.02,2.99,3.11]
 diff=[b-a for a,b in zip(base,candidate)]
 result={'scope':'synthetic observations, not trained model scores','baseline':base,'candidate':candidate,
 'paired_differences':diff,'mean_difference':statistics.mean(diff),
 'sample_standard_deviation':statistics.stdev(diff),'standard_error':statistics.stdev(diff)/math.sqrt(len(diff)),
 'factorial_interaction':2.7-3.1-3.0+3.2}
 s=json.dumps(result,ensure_ascii=False,indent=2)
 if a.output:a.output.write_text(s+'\n')
 print(s)
if __name__=='__main__':main()
