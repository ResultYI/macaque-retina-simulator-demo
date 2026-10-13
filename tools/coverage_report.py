from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retina_sim.provenance import validate_runnable
ROOT=Path(__file__).resolve().parents[1]
items=[
('P01','species/source policy','POLICY_CONFIRMED',[],['用户共同框架政策已确认；具体猕猴参数仍需分物种/来源补齐'],'Human/cat excluded from main defaults; author mixture opt-in only'),
('S01','continuous centre/transition/periphery model','BLOCKED',[],['全域机制与连续空间 profile 未冻结'],'No complete retina claim'),
('S02','nonuniform L/M/S cone mosaic','BLOCKED',['ISETMacaque_PackerConeRc'],['e=0、垂直 meridians、完整二维及 L/M/S/S 区域规则未核齐'],'60 meridian nodes are not a 2-D mosaic'),
('S03','calibrated light to R*/s','BLOCKED',[],['optics、pupil、显示光谱和真实效率 profile 未闭合'],'Unit operator tested only with synthetic_fixture'),
('S04','horizontal cells and feedback','AVAILABLE_REFERENCE',['SF2025_runColFun_m'],['主线 HC 生理参数/机制未冻结'],'Named H1 linear reference does not unblock main model'),
('S05','ON/OFF bipolar types, nonlinearity and wiring','BLOCKED',['SF2025_runColFun_m'],['完整 subtype/非线性和连接证据未齐'],'Midget linear reference only'),
('S06','amacrine types, states and targets','BLOCKED',[],['AC 类型、状态、靶细胞和连接未知'],'No AC implementation or substitute'),
('S07','RGC type coverage and spike mechanism','BLOCKED',['SF2025_runColTask_m'],['其他 RGC 类型和需要的 spike mechanism 未冻结'],'Midget rate only; no invented stochastic spikes'),
('S08','anatomical/functional displacement and Jacobian','BLOCKED',[],['位移模型、Jacobian、计数守恒未核齐'],'Separate nullable coordinates are not a displacement model'),
('S09','sparse directed connectivity','ENGINEERING_VALIDATED',[],[],'CSR postsynaptic rows; source-free edges cannot enter physiological runs'),
('S10','regional dynamics and continuous profile','BLOCKED',[],['中心—周边动态函数未知；区域端点不能自动插值为生理机制'],'No default regional interpolation'),
('S11','given gaze and persistent state','BLOCKED',[],['实际眼动到视网膜图像的校准映射未闭合'],'Streaming state tested; calibrated gaze pathway absent'),
('S12','layer outputs with units and metadata','ENGINEERING_VALIDATED',['SF2025_runColFun_m'],[],'Reference layer ports; not all physiological cell classes'),
('S13','independent physiological validation','NOT_RUN',[],['独立验证数据与预先指标未登记；训练/生理实验本阶段未运行'],'Numerical fixtures never count as physiological observations'),
('S14','separate numerical, coverage and physiology reports','ENGINEERING_VALIDATED',[],[],'Evidence reports do not certify biology'),
('E01','source/parameter/unit gates','ENGINEERING_VALIDATED',[],[],'Schema plus semantic/profile/asset checks; unknown full parameters rejected'),
('E02','cone density primary evidence','PARTIAL_SOURCE',['ISETMacaque_PackerConeRc'],['原始测量归属、物种个体、数字化误差未知'],'published_code_tabulation, not measured_source_table'),
('A01','physiological cone aperture','BLOCKED',[],['真实采光面积与空间 aperture profile 未核齐'],'Aperture unit interface is not a biological parameter'),
('A02','phototransduction and nonlinear adaptation','BLOCKED',[],['光转导/calcium/bleaching/Ih 与背景依赖适应未闭合'],'Saha fits not converted into an unknown ODE'),
('B01','author arrays and paper reproduction','BLOCKED_DATA_DECODE',['SF2025_Cone_density_mat','SF2025_Ganglion_cell_density_mat'],['两个 MCOS table 未解码数字与列名'],'Source-faithful kernel only, no original-array/paper reproduction')]
evidence_gaps={'P01': ['逐物种、区域与条件须各自核验，不以混合来源替代主线默认'], 'S01': ['中心、过渡和周边尚无完整生理 profile'], 'S02': ['e=0、各 meridian、真实二维 L/M/S mosaic 与 S cone 参数缺失'], 'S03': ['标定 optics、pupil、光谱与区域输入参数未具备'], 'S04': ['完整 HC 类型、非线性反馈与连接证据缺失'], 'S05': ['ON/OFF BC subtype、非线性和连接参数缺失'], 'S06': ['AC 类型、状态与目标连接尚未实现'], 'S07': ['完整 RGC 类型与 spike 生理机制缺失'], 'S08': ['生理位移模型、Jacobian 和区域坐标参数缺失'], 'S09': [], 'S10': ['区域依赖动力学与连续参数过渡缺失'], 'S11': ['给定眼动到标定视网膜光输入的完整映射缺失'], 'S12': [], 'S13': ['独立、未用于定参数的生理验证数据与条件缺失'], 'S14': [], 'E01': [], 'E02': ['真实二维密度、位置、物种和条件证据仍不完整'], 'A01': ['生理 aperture profile 与来源参数缺失'], 'A02': ['中心、过渡、S cone、电压耦合、terminal Ca/HC，以及真实来源对照和独立验证缺失'], 'B01': ['Somaratna–Freeman 作者 MCOS 数值表未解码']}
requirements=[]
for rid,module,gate,sources,missing,limit in items:
    status='数值验证' if gate=='ENGINEERING_VALIDATED' else '局部实现' if gate=='POLICY_CONFIRMED' else '来源待补'
    missing=evidence_gaps[rid]
    requirements.append(dict(requirement_id=rid,required_module=module,status=status,gate=gate,implemented_artifact='retina_sim' if gate in ('ENGINEERING_VALIDATED','AVAILABLE_REFERENCE','POLICY_CONFIRMED') else None,source_ids=sources,missing_evidence=missing,tests=['tests/test_stage_a.py','tests/test_stage_b.py'],claim_limit=limit))
for r in requirements:
    if r['requirement_id']=='A02' and (ROOT/'reports/chen2024_stage_c_status.json').is_file():
        c=json.loads((ROOT/'reports/chen2024_stage_c_status.json').read_text(encoding='utf-8'))
        r['implemented_artifact']='retina_sim/chen2024.py'
        r['source_ids']=['CHEN2024_BiophysModel_m','CHEN2024_initPhotoreceptorParams_m']
        r['subrequirements']=[c]
        r['missing_evidence']=evidence_gaps['A02']
        inventory=ROOT/'reports/chen2024_data_inventory.json'
        if inventory.is_file():
            actual=json.loads(inventory.read_text(encoding='utf-8'))
            if actual.get('identity_and_structure')=='VERIFIED':
                r['subrequirements']+=[{'requirement_id':'A02.source_data_identity','status':'来源资产与结构已核验','gate':'EVIDENCE_IDENTITY_VERIFIED','implemented_artifact':'reports/chen2024_data_inventory.json','source_ids':['CHEN2024_Dryad_SourceData','CHEN2024_Dryad_README'],'claim_limit':'Published byte identity and real fields only; no model-data fit claim'},{'requirement_id':'A02.source_data_reproduction','status':'来源待补','gate':actual['status'],'implemented_artifact':'docs/chen2024_real_data_report_zh.md','missing_evidence':['Negative-rate meaning/recovery; prehistory/phase mapping'],'claim_limit':'Model comparison NOT_RUN; high-light applicability remains a biological warning; no clipping, fitting, time shift or invented state'}]
(ROOT/'requirements.json').write_text(json.dumps(requirements,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=validate_runnable({'mode':'full_physiology'},requirements)
(ROOT/'reports/full_mode_blockers.json').parent.mkdir(exist_ok=True)
(ROOT/'reports/full_mode_blockers.json').write_text(json.dumps(report.to_dict(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 完整需求覆盖表','', '状态分为未开始、局部实现、来源待补、数值验证、生理验证；gate 控制完整模式。','', '| ID | 模块 | 状态 | Gate | 缺失证据 / 限制 |','|---|---|---|---|---|']
for r in requirements:
    lines.append('| '+ ' | '.join([r['requirement_id'],r['required_module'],r['status'],r['gate'],'；'.join(r['missing_evidence'])+' '+r['claim_limit']])+' |')
lines+=['','完整模式阻塞见 reports/full_mode_blockers.json。数值检查通过不认证生理符合度；本表保留所有完整主线需求。']
if (ROOT/'reports/chen2024_stage_c_status.json').is_file():
    lines+=['','## A02 外周 L/M 子项','', 'Chen2024 外周 >20deg L/M：数值验证 / SOURCE_PARITY_VALIDATED。A02 父项仍 BLOCKED；真实数据身份、字段与当前对照阻塞见 reports/chen2024_data_inventory.json 和 docs/chen2024_real_data_report_zh.md。']
(ROOT/'docs/requirements_coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'requirements':len(requirements),'full_mode_allowed':report.allowed,'blockers':len(report.blockers)}))
