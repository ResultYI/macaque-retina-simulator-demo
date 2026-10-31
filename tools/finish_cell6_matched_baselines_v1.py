"""Deliver saved matched-input results with the existing plotting interpreter."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def finish(out):
    complete=read(out/'COMPUTATION_COMPLETE.json')
    frozen=read(out/'MODELS_FROZEN.json');frozen_hash=sha(out/'MODELS_FROZEN.json')
    if complete['groups']!=6 or complete['retinal_model_solves']!=0:raise RuntimeError('Incomplete comparison')
    for name,expected in frozen['source_pins'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Source pin changed: '+name)
    for group in range(6):
        result=read(out/f'g{group}_RESULT.json');identity=read(out/f'g{group}_INPUT_IDENTITY.json')
        if result['models_frozen_sha256']!=frozen_hash:raise RuntimeError('Model freeze differs')
        if not identity['full_program_bitwise_identity_closed'] or not identity['saved_native_anchor_light_exact']:
            raise RuntimeError('Input identity is not closed')
        if identity['T_program_sha256']!=identity['reconstructed_program_sha256']:
            raise RuntimeError('Matched input program differs')
    from tools.deliver_cell6_matched_baselines_v1 import deliver
    deliver(out)
    comparisons=read(out/'PAIRED_COMPARISONS.json')
    lines=['# 同输入补充比较结论','',
        '六组输入均已用 T 存储的帧索引、实际时钟和 Cone 光输入 SHA 闭合；没有重跑视网膜网络。',
        '仅刺激改为 T 实际使用的 150 Hz 保持场景，LN/GLM 原基函数、历史、正则候选、组内选择和随机数规则保持。',
        '所有参数只由 g0–g2 训练并冻结，随后 g3–g5 固定评估一次。v1 原始输入版及原 T 结果均保留。','',
        '以下结论逐指标判断，不能据此证明网络结构的因果贡献。GLM 200 ms 历史与 T 3 ms 历史仍不相同。']
    for group,models in comparisons.items():
        for name,values in models.items():
            lines.append(f'- g{group} T 对 {name}：'+json.dumps(values['metric_conclusions'],ensure_ascii=False))
    lines+=['','g3 为已触碰内部检查，g4/g5 为历史开发数据；不是独立未触碰测试。历史 test 未访问。',
        '原 g5 INCONCLUSIVE 不变。FIT_READY/background_qualified/production_adopted 均为 false。',
        '完整指标与区间见 FINAL_REPORT.md、COMPARISON.csv、PAIRED_COMPARISONS.json；曲线见 DEVELOPMENT_OVERLAY.png/svg。']
    (out/'CONCLUSION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(out/'DELIVERY.json',dict(interpreter=sys.executable,saved_arrays_only=True,
        new_fits=0,new_predictions=0,new_evaluations=0,input_programs_matched=6,
        utc=datetime.now(timezone.utc).isoformat()))
    write(out/'COMPLETE.json',dict(complete,status='COMPLETE_MATCHED_INPUT_SUPPLEMENT',
        original_v1_preserved=True,input_programs_bitwise_matched=6,
        model_freeze_sha256=frozen_hash,FIT_READY=False,background_qualified=False,production_adopted=False))


if __name__=='__main__':finish(Path(sys.argv[1]))
