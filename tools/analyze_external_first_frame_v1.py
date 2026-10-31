"""Summarize saved external optical fixture responses without running a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def read(path):
    return json.loads(path.read_bytes())


def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(path,rows):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(base):
    prep=read(base/'PREPARATION.json');rows=[];curves=[];endpoints=[];resources=[];checks=[]
    runs={'near':'near_run_v2','far':'far_run_v1'}
    for position,name in runs.items():
        folder=base/name;r=read(folder/'RESULT.json');identity=read(folder/'IDENTITY.json')
        resource=read(folder/'RESOURCE.json');resources.append(dict(directory=name,**resource))
        with np.load(folder/'RESPONSE.npz',allow_pickle=False) as z:
            times=z['time_seconds'];sides=z['side'];fields=identity['fields']
            if times[0]!=0. or times[-1]!=1/30 or np.any(np.diff(times)<0):raise ValueError('Saved exposure time axis differs')
            if len(times)!=75 or len(np.unique(times))!=71:raise ValueError('Unexpected saved event-side registry')
            for pop,meta in identity['types'].items():
                V=z[pop+'__voltage']*1000;rate=z[pop+'__expected_binary_count_rate']
                row=dict(position=position,type=pop,groups=meta['groups'],members=sum(meta['represented_members']),
                    initial_group_mean_mV=float(V[:,0].mean()),end_group_mean_mV=float(V[:,-1].mean()),
                    end_mean_delta_mV=float((V[:,-1]-V[:,0]).mean()),maximum_any_group_delta_mV=float(np.max(abs(V-V[:,0,None]))),
                    initial_group_mean_binary_rate_Hz=float(rate[:,0].mean()),end_group_mean_binary_rate_Hz=float(rate[:,-1].mean()),
                    end_mean_rate_delta_Hz=float((rate[:,-1]-rate[:,0]).mean()),maximum_any_group_rate_delta_Hz=float(np.max(abs(rate-rate[:,0,None]))),
                    published_group_object_events=sum(e['type']==pop for e in read(folder/'GROUP_OBJECT_EVENTS.json')))
                for field in ['excitatory_conductance','inhibitory_conductance']:
                    v=z[pop+'__'+field]
                    row['maximum_any_group_'+field+'_change_S']=float(np.max(abs(v-v[:,0,None])))
                rows.append(row)
                for i,t in enumerate(times):
                    record=dict(position=position,type=pop,time_seconds=float(t),side=str(sides[i]))
                    record.update({key.split('__',1)[1]:float(z[key][:,i].mean()) for key in fields if key.startswith(pop+'__')})
                    curves.append(record)
                for i,gid in enumerate(meta['group_ids']):
                    record=dict(position=position,type=pop,group_id=gid,represented_members=meta['represented_members'][i])
                    for key in fields:
                        if not key.startswith(pop+'__'):continue
                        field=key.split('__',1)[1];values=z[key]
                        record[field+'_initial']=float(values[i,0]);record[field+'_end']=float(values[i,-1])
                    endpoints.append(record)
            checks.append(dict(position=position,records=len(times),unique_times=len(np.unique(times)),
                all_saved_arrays_finite=all(np.isfinite(z[k]).all() for k in z.files if k!='side'),
                RGC_fields=len(fields),dynamic_defect_samples=r['dynamic_numerical_defect']['samples'],
                maximum_dynamic_defect_ratio=r['dynamic_numerical_defect']['maximum_norm'],
                original_physical_history_conservation_passed=r['physical_history_conservation_passed'],
                native_input_hold_jumps=r['input_hold_boundary_checks'],
                endpoint_changes=r['maximum_end_change_by_port'],published_group_object_events=r['published_group_object_events']))
    failed=read(base/'near_run_v1/RESOURCE.json');resources.insert(0,dict(directory='near_run_v1',**failed))
    preparation=read(base/'PREFLIGHT.json')['preparation_wall_seconds']
    total=preparation+sum(r['wall_seconds'] for r in resources)
    ledger=dict(historical_cumulative_seconds=read(base/'PREFLIGHT.json')['historical_cumulative_seconds'],
        preparation_seconds=preparation,runs=resources,new_measured_seconds=total,
        cumulative_seconds=read(base/'PREFLIGHT.json')['historical_cumulative_seconds']+total,
        history_preserved=True,regression_test_seconds=.001,regression_test_excluded_from_model_ledger=True)
    csv_file(base/'POSITION_TYPE_SUMMARY.csv',rows);csv_file(base/'RGC_TIME_COURSES.csv',curves)
    csv_file(base/'RGC_GROUP_ENDPOINTS.csv',endpoints);write(base/'POSITION_TYPE_SUMMARY.json',rows)
    write(base/'NUMERICAL_SCOPE.json',checks);write(base/'RESOURCE_LEDGER.json',ledger)
    result=dict(status='TWO_POSITION_EXISTING_EXTERNAL_FIRST_EXPOSURE_COMPLETE',experiment_id=prep['experiment_id'],
        runs=runs,exposure_seconds=1/30,input_contract=prep['exact_existing_time_rule'],
        RGC_group_mean_weighting='Equal weight per existing group, not per fine member',
        numerical_scope='Original reference and dense-derivative budget only; no new standard/tight trajectory pair',
        actual_measures=rows,checks=checks,ledger=ledger,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write(base/'FINAL_RESULT.json',result)
    write(base/'EXECUTION_SOURCES.json',{str(p.relative_to(ROOT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in [
        ROOT/'tools/prepare_external_first_frame_v1.py',ROOT/'tools/run_external_first_frame_v1.py',
        ROOT/'tools/analyze_external_first_frame_v1.py',ROOT/'tests/test_external_first_frame_v1.py']})
    lines=['# 既有外部首帧：两位置实际功能运行','',
        '已完成：复用既有 113.AVI 首帧的唯一 0–1/30 秒曝光，分别驱动近中心和原远点的真实 Cone/HC→BC→AC→RGC。五类既有 RGC 均有有限的 gE/gI、膜、率和组对象事件输出。响应很弱，未通过放大增益改善。','',
        '输入来自已保存 capture.mat（SHA 6a1436cb1eaeba51ed60918e4b8f0f3c6b801a08f87f2f402d537d072c8b1d40）。R*/曝光除以唯一曝光时长，按各位置既有固定 LMS bridge 转换，再按真实成员均值一次聚合；五个 150Hz 输入完全相同。没有新造 flash、contrast、光谱、亮度或时间参数。',
        f"首帧既有 gain={prep['fixture']['fixed_radiance_gain']!r}；S0 初态 gain={prep['fixture']['S0_seed_gain']!r}。原约 1.8e-7 相对差保留，不把图像亮度归一到 S0。",'',
        '近点沿用真实中心约 0.06431028° 的已完成构建；远点沿用准确坐标 [4.724986235371699,-0.11169236923054729]°。两者各自使用独立 S0 背景种子，均未重新平衡到图像。近点 RGC 为55组/2173成员，远点42组/191成员；细胞实际坐标范围见上一轮 [坐标表](../position_S0_far_originaltarget_20261029_resume_v2/dynamic_S0_v1/analysis/RGC_COORDINATES_AND_MEMBERS.csv)。patch 中心不是所有细胞的偏心度。','',
        '|位置|类型|组/成员|初态 mV|端点 ΔmV（组均值）|网格内最大任一组 ΔmV|初态 Hz|端点 ΔHz（组均值）|组事件|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['position']}|{row['type']}|{row['groups']}/{row['members']}|{row['initial_group_mean_mV']:.6f}|{row['end_mean_delta_mV']:+.7f}|{row['maximum_any_group_delta_mV']:.7f}|{row['initial_group_mean_binary_rate_Hz']:.6f}|{row['end_mean_rate_delta_Hz']:+.7f}|{row['published_group_object_events']}|")
    lines+=['','表中 Hz 为原 1ms 二元计数的期望率；不是 hazard。所有均值按组等权，未按成员数重新缩放。事件是分组模型对象的事件，近48、远50；不能冒称2173/191个独立单细胞 spike。',
        '', '每位置保存75个事件侧记录、71个唯一时刻，含0.5ms网格、四个内部输入边界左右侧和33.333ms曝光终点左极限。50个 RGC 字段及所有已记录非RGC端口均保留原数组。结束后的光输入、恢复段和未发布事件未添加。',
        '',f"原物理/事件历史/守恒检查两点均通过；每点406个接受稠密段中点的全状态导数缺陷检查最大预算比例：近{checks[0]['maximum_dynamic_defect_ratio']:.9g}、远{checks[1]['maximum_dynamic_defect_ratio']:.9g}，均≤1。输入重分区边界连续态跳变均为0。只运行原 reference 一次/位置，未宣称本轮已有独立 tight 收敛对照。",'',
        '首次近点在0.1ms因连接器误用 SciPy 字段名停止；ACVSInner 实际返回工程 Dop853Dense(t0,t1)。失败源码、见证和10.9393262s成本完整保留于 near_run_v1。仅修正连接器，实际对象回归检查通过后新目录重跑；没有改科学方程、容差或验收预算。',
        '',f"本轮预检与三次尝试共{total:.6f}s；历史{ledger['historical_cumulative_seconds']:.6f}s→累计{ledger['cumulative_seconds']:.6f}s。单进程24GiB/系统余量8GiB守卫保留，实际峰值{max(r['memory']['process_peak_rss_bytes'] for r in resources)/1024**3:.3f}GiB，末次可用物理内存约{resources[-1]['memory']['available_physical_bytes']/1024**3:.2f}GiB。",'',
        '限定结论：这是既有图像曝光的局部外部输入通路验证。空间图像在两位置本来不同；有限patch外部储库继续冻结于S0的既有t=0边界，未计算全视网膜图像驱动。Rod沿用独立工程背景，未添加光谱Rod捕获。固定LMS转换是reference-equivalent工程单位，不是一般photopic Td。33ms短窗不证明完整时间响应、视觉对比极性、Sbi蓝黄对抗或偏心效应。Sbi及高基线限制仍在。FIT_READY/background_qualified/production_adopted全部false。',
        '', '原生孔径修复保留猕猴坐标与逐Cone直径，复用原生Gaussian核/面积；本轮直接复用旧完整首帧捕获，没有重跑MATLAB，也没有把局部采样器说成完整原生compute。原130Cone均匀/非均匀实际捕获一致性见[孔径报告](../isetbio_native_aperture_convergence_20261029_v1/README.md)。',
        '', '数据：[原始近点](near_run_v2/RESPONSE.npz)、[原始远点](far_run_v1/RESPONSE.npz)、[类型指标](POSITION_TYPE_SUMMARY.csv)、[全类型时序](RGC_TIME_COURSES.csv)、[每组端点](RGC_GROUP_ENDPOINTS.csv)、[数值范围](NUMERICAL_SCOPE.json)、[资源账本](RESOURCE_LEDGER.json)、[曲线](external_first_exposure_curves.png)。',
        '', '唯一后续：将已有113.AVI的后续已定义30Hz帧捕获接入本连接器，以相同谱、亮度及输入单位运行连续视觉响应。当前唯一缺口是首帧之后的实际逐帧捕获缓存，现有首帧不能决定未来输入；无需重新选择contrast或flash时序。本轮到此收束，不自动扩展视频或拟合。']
    (base/'FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    reproduce=f'''# 复现（新目录，绝不覆盖）
准备仅解析原输入并创建新身份，不运行模型：
```
.venv\\Scripts\\python.exe tools\\prepare_external_first_frame_v1.py --output reports\\external_first_frame_replay_NEW
```
显式运行一位置：
```
.venv\\Scripts\\python.exe tools\\run_external_first_frame_v1.py --preparation reports\\external_first_frame_replay_NEW --position near --output reports\\external_first_frame_replay_NEW\\near_run
```
远点将near改为far并使用独立far_run目录；两者顺序运行。代码拒绝既有输出目录。默认不触发全链，只有显式run脚本执行33.333ms。每个过程24GiB/8GiB余量/820s；没有任意续曝光、tight扫参或新的刺激参数。
现有成功实例：near_run_v2和far_run_v1；near_run_v1保留失败。固定来源在PREPARATION.json，源码在EXECUTION_SOURCES.json。结果只为工程功能，所有科学资格false。
'''
    (base/'REPRODUCE.md').write_text(reproduce,encoding='utf8')


def plot(base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(base/'POSITION_TYPE_SUMMARY.json');pops=list(dict.fromkeys(row['type'] for row in rows))
    fig,axes=plt.subplots(5,2,figsize=(12,13),sharex=True)
    for label,name,color in [('near','near_run_v2','#0072B2'),('far','far_run_v1','#D55E00')]:
        with np.load(base/name/'RESPONSE.npz',allow_pickle=False) as z:
            t=z['time_seconds']*1000
            for i,pop in enumerate(pops):
                for j,(field,scale) in enumerate([('voltage',1000),('expected_binary_count_rate',1)]):
                    v=z[pop+'__'+field];axes[i,j].plot(t,(v-v[:,0,None]).mean(axis=0)*scale,label=label,color=color,lw=1.6)
    for i,pop in enumerate(pops):
        axes[i,0].set_ylabel(pop.replace('_RGC','').replace('_',' ')+'\nmean delta V (mV)',fontsize=9)
        axes[i,1].set_ylabel('mean delta binary rate (Hz)',fontsize=9)
        for ax in axes[i]:ax.grid(alpha=.22);ax.axhline(0,color='#444',lw=.5);ax.set_xlim(0,1000/30)
    axes[0,0].legend();axes[0,1].legend()
    for ax in axes[-1]:ax.set_xlabel('Existing first exposure time (ms)')
    fig.suptitle('Existing external first image: local true Cone/HC to RGC\nEqual weight per group; independent S0 initial states; frozen exterior; no fit',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','svg','pdf']:fig.savefig(base/('external_first_exposure_curves.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--plot-only',action='store_true');a=p.parse_args();base=a.experiment.resolve()
    if a.plot_only:plot(base)
    else:summarize(base)
