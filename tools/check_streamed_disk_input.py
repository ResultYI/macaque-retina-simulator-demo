"""One tiny actual-ISET eager/streamed comparison on existing QA footage."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from pytest import MonkeyPatch
from test_offline_input import fixture
from test_streamed_input import exercise_frontend_pair

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use a fresh project QA directory')
    output.mkdir(parents=True);tick=time.perf_counter()
    with MonkeyPatch.context() as patch:
        result=exercise_frontend_pair(output,fixture.__wrapped__(),patch,real_iset=True)
    result['wall_seconds']=time.perf_counter()-tick
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
