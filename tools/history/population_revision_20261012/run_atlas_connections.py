"""Bounded representative connection geometry; never a background experiment."""
from pathlib import Path
import sys,json,argparse
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_platform.atlas_connections_preview import export_connection_previews

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('strict','bridge_strict_S','bridge_mulatta_effective'),default='strict')
    args=parser.parse_args()
    cases=[{'label':'中心 RF r=0：缺 IS→OPL 起点','center_mm':[0,0],'size_mm':[.008,.008]},
        {'label':'中心邻域 RF r=.035：主 S-mask 约定，低证据 HC 候选','center_mm':[.035,0],'size_mm':[.004,.004],'halo_mm':.007},
        {'label':'鼻侧 RF r=.35：已支持 OPL，S 分区未知','center_mm':[.35,0]},
        {'label':'过渡鼻侧 RF r=1：有限外围插值候选','center_mm':[1,0]},
        {'label':'过渡上方 RF r=1：角向迁移明确为近似','center_mm':[0,1]},
        {'label':'边缘 RF r=3：Henle 尾段缺；独立 OPL r=3 HC 查询','center_mm':[3,0],
         'HC_reference_bounds_OPL_mm':[2.95,3.05,-.05,.05]}]
    if args.variant!='strict':
        cases[0]['label']='中心 RF r=0：批准中央P桥接，原始来源版独立保留'
        cases[-1]['label']='边缘 RF r=3：批准尾段P桥接；HC master未物化'
    if args.variant=='bridge_mulatta_effective':
        cases[1]['label']='中心邻域 RF r=.035：mulatta有效S；HC footprint候选'
        cases[2]['label']='鼻侧 RF r=.35：mulatta有效S；完整HC输入查询'
    path=ROOT/'reports'/'atlas_connections'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result=export_connection_previews(ROOT,path,cases,
        atlas_profile_id='retina_atlas_v1' if args.variant=='strict' else 'retina_atlas_Schein_bridge_research_v1',
        s_partition_profile='deMonasterio1985_mulatta_effective_S_v1' if args.variant=='bridge_mulatta_effective' else 'nemestrina_strict_pending_v1')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
