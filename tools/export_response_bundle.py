"""Export a completed run into one portable, classified desktop response ZIP."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_platform.response_bundle import open_result,export_package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    source=open_result(args.experiment,ROOT)
    try:
        target=args.output or ROOT/'reports/result_packages'/('retina_113_4s_'+source.manifest['run_id'][:12]+'.response.zip')
        result=export_package(source,target,lambda text:print(text,flush=True))
        print(json.dumps(dict(status='COMPLETE_PORTABLE_RESPONSE',package=str(result),model_run=False)),flush=True)
    finally:source.close()


if __name__=='__main__':main()
