"""Build the canonical Chinese model description from its manuscript and source snapshot."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'document'
MANUSCRIPT = OUT / '当前建模原理.md'
SNAPSHOT = OUT / '当前建模参数与来源.json'
PDF = OUT / '猕猴视网膜模拟器_当前建模原理.pdf'


def refresh_snapshot():
    extraction = r'''
import json,hashlib
from pathlib import Path
import numpy as np
from retina_sim.group_time_storage import load_bundle
r=Path.cwd()
base=r/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source'
s=load_bundle(base/'seed',expected_kind='source_construction_seed')['payload']
def summarize(x):
 if isinstance(x,dict):return {str(k):summarize(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [summarize(v) for v in x] if len(x)<20 else dict(count=len(x),meaning='see original referenced array')
 if isinstance(x,np.ndarray):
  if x.size<=8:return x.tolist()
  if x.dtype.kind in 'biuf':return dict(shape=list(x.shape),min=float(x.min()),max=float(x.max()))
  return dict(shape=list(x.shape),dtype=str(x.dtype))
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 return x
p=s['inner_parameters']
active={k:v for k,v in p['populations'].items() if k not in ('local_crossover_AC','widefield_AC')}
sections=dict(populations=summarize(active),route_gains=summarize(p['route_gains']),rod_phototransduction=summarize(p['rod_phototransduction']),TRPM1=summarize(p['contacts']['individual_rod_RBC_contacts']))
sources=[
'configurations/research/AC_functional.json','configurations/research/BC_macaque_fullchain.json',
'configurations/research/cone_hc_h2_priority.json','configurations/research/cone_hc_eccentric.json',
'configurations/research/HC_eccentric_biexponential.json','configurations/research/H2_S_priority.json',
'configurations/ac_vs_functional_v1.json','configurations/hc_density_rf_joint_v2_owner_support_v1.json',
'configurations/bc_density_geometry_current.json','configurations/bc_cone_pedicle_anchored_owner_support_v1.json',
'configurations/research/GIN69_OWNER_SUPPORT_V2.json',
'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json',
'data/atlas/isetbio_human_borrowed_cones_disk3mm_v1/manifest.json',
'retina_sim/population_revision.py','retina_sim/disk_inner.py','retina_sim/isetbio_input.py',
'retina_sim/hc_lateral_candidate.py',
'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/HC_PARAMETER_LEDGER.json',
'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/CALIBRATION_METRICS.json',
'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/H2_SPECTRAL_AUDIT.json',
'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02/FINAL_REPORT.md',
'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02/BC_PARAMETER_LEDGER.json',
'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02/CALIBRATION_METRICS.json',
'data/source_data/bc_physiology_targets.json','data/source_data/h1_temporal_targets.csv',
'reports/ac_vs_redesign_v1/FINAL_REPORT.md','reports/ac_vs_redesign_v1/FUNCTIONAL_PARAMETERS.json',
'reports/ac_vs_redesign_v1/FUNCTIONAL_ACCEPTANCE.json','reports/ac_vs_redesign_v1/LAYERED_CALIBRATION.json',
'reports/ac_vs_redesign_v1/LAYERED_PARAMETERS.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source/BACKGROUND.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source/STOPPED_BY_USER.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source/seed/manifest.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/SPLIT_CONTRACT.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/input_policy.json',
'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/joint_native_prefix_01/RESULT.json',
'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2/INPUT_APPROXIMATION_MANIFEST.json',
'sources/core_registry.json','sources/registry.json','sources/primary_pathway_effective_profile.json',
'retina_sim/vh2005.py','retina_sim/hc_eccentric.py','retina_sim/h1_h2_effective.py',
'retina_sim/h2_priority.py','retina_sim/bc_independent.py','retina_sim/bc_gain_control.py',
'retina_sim/rod_pathway.py','retina_sim/chen2024.py','retina_sim/group_inner_dynamics.py',
'retina_sim/ac_vs_model.py','retina_sim/aii_functional.py','retina_sim/rgc_expectation.py',
'tools/run_gin69_current_ac.py','tools/gin69_voltage_fit_support_v2.py',
'tools/build_model_document.py',
'document/当前建模原理.md','document/项目整理记录.json']
bindings=[dict(path=f,sha256=hashlib.sha256((r/f).read_bytes()).hexdigest()) for f in sources]
bg=json.loads((base/'BACKGROUND.json').read_bytes())
snap=dict(date='2026-11-04 Asia/Tokyo',scope='current research AC/AII model and paused 69#6 input; no simulation or fitting',
model_identity=bg['model_identity'],source_identity=bg['identity'],rng_seed=summarize(s['rng_seed']),
parameters=sections,H2=summarize(s['h1_h2_model']['parameters']),
optical_aperture_binding=summarize(s.get('optical_aperture_input')),
source_files=bindings,full_array_policy='Summaries do not replace existing per-cell/edge arrays; seed manifest remains the complete source')
print(json.dumps(snap,ensure_ascii=False))
'''
    result = subprocess.run(
        [r'D:\anaconda\envs\snn_env\python.exe', '-B', '-X', 'utf8', '-c', extraction],
        cwd=ROOT, check=True, capture_output=True, encoding='utf-8')
    data = json.loads(result.stdout)
    SNAPSHOT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def build():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                   Spacer, PageBreak, Table, TableStyle, KeepTogether)
    from reportlab.platypus.tableofcontents import TableOfContents

    pdfmetrics.registerFont(TTFont('CJK', r'C:\Windows\Fonts\msyh.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKB', r'C:\Windows\Fonts\msyhbd.ttc', subfontIndex=0))
    pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB', italic='CJK', boldItalic='CJKB')
    ink = colors.HexColor('#17334A')
    body = ParagraphStyle('body', fontName='CJK', fontSize=10, leading=15.5,
                          textColor=colors.HexColor('#283745'), spaceAfter=6.5, rightIndent=7,
                          wordWrap='CJK', splitLongWords=True, allowWidows=0, allowOrphans=0)
    h1 = ParagraphStyle('h1', parent=body, fontName='CJKB', fontSize=18, leading=25,
                        textColor=ink, spaceAfter=14, keepWithNext=True)
    h2 = ParagraphStyle('h2', parent=body, fontName='CJKB', fontSize=12, leading=19,
                        textColor=ink, spaceBefore=9, spaceAfter=7, keepWithNext=True)
    cell = ParagraphStyle('cell', parent=body, fontSize=8.3, leading=12, spaceAfter=0, rightIndent=0)
    small = ParagraphStyle('small', parent=body, fontSize=8.2, leading=12.5)
    eq = ParagraphStyle('eq', parent=body, fontSize=9.2, leading=15.3, leftIndent=10,
                        rightIndent=8, borderPadding=8, backColor=colors.HexColor('#EFF4F7'))
    title = ParagraphStyle('title', parent=h1, fontSize=30, leading=43, spaceAfter=22)
    subtitle = ParagraphStyle('subtitle', parent=h2, fontSize=17, leading=27, spaceAfter=30)
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('toc', parent=body, fontSize=10, leading=18, spaceAfter=4)]
    width = A4[0] - 92

    def escaped(text):
        text = html.escape(str(text))
        text = re.sub(r'`([^`]+)`', r'<font color="#185B78">\1</font>', text)
        # Break Latin paths and long identifiers in a CJK paragraph without altering extracted text.
        return text

    def p(text, style=body):
        return Paragraph(escaped(text), style)

    def table(rows, widths=None):
        n = len(rows[0])
        if widths is None:
            if n == 2: widths = [width*.34, width*.66]
            elif n == 3: widths = [width*.26, width*.42, width*.32]
            elif n == 4: widths = [width*.25]*4
            else: widths = [width/n]*n
        data = [[p(x, cell) for x in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DFEAF0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F6F8FA')]),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#A6BDCB')),
            ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#CBD8E0')),
        ]))
        return t

    class Document(BaseDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name == 'h1' and flowable.getPlainText() != '阅读导航':
                text = flowable.getPlainText()
                key = hashlib.sha1(text.encode()).hexdigest()
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, 0)
                self.notify('TOCEntry', (0, text, self.page, key))

    def page(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont('CJK', 8)
            canvas.setFillColor(colors.HexColor('#708391'))
            canvas.drawString(46, A4[1]-30, '猕猴视网膜模拟器 | 当前建模原理与证据')
            canvas.setStrokeColor(colors.HexColor('#D8E3EA'))
            canvas.line(46, A4[1]-38, A4[0]-46, A4[1]-38)
        canvas.setFont('CJK', 8)
        canvas.setFillColor(colors.HexColor('#708391'))
        canvas.drawString(46, 26, '研究模型说明 · 2026-11-04 · 非全模块生理标定声明')
        canvas.drawRightString(A4[0]-46, 26, str(doc.page))
        canvas.restoreState()

    doc = Document(str(PDF), pagesize=A4, leftMargin=46, rightMargin=46,
                   topMargin=55, bottomMargin=47,
                   title='猕猴视网膜模拟器：当前建模原理、参数与证据边界',
                   author='项目建模说明', pageCompression=1)
    doc.addPageTemplates(PageTemplate(id='main', frames=[Frame(46,47,width,A4[1]-102,
                         leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)], onPage=page))
    lines = MANUSCRIPT.read_text(encoding='utf-8').splitlines()
    story = [Spacer(1,50)]
    i = 0
    cover = True
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line == '<!-- appendix -->':
            break
        if line == '<!-- page -->':
            story.append(PageBreak())
            if cover:
                story.extend([p('阅读导航', h1), toc, PageBreak()])
                cover = False
            i += 1; continue
        if line.startswith('# '):
            story.append(p(line[2:], title if cover else h1)); i += 1; continue
        if line.startswith('## '):
            story.append(p(line[3:], subtitle if cover else h2)); i += 1; continue
        if line.startswith('```'):
            values = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                values.append(escaped(lines[i])); i += 1
            story.extend([Paragraph('<br/>'.join(values), eq), Spacer(1,8)])
            i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', x) for x in row): rows.append(row)
                i += 1
            story.extend([table(rows), Spacer(1,10)]); continue
        story.append(p(line)); i += 1

    snap = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    story.extend([PageBreak(), p('附录 A. 当前绑定参数摘要', h1),
        p('以下直接读取暂停任务的 source_construction_seed。单值为实际绑定值；shape/min/max 为原数组摘要，单位沿用源码 SI 合同。旧 AII 化学受体参数与未使用 AC 池不列为活动参数。完整逐行值保留在原 seed 引用数组中。')])

    def value(v):
        if isinstance(v,float): return f'{v:.10g}'
        if isinstance(v,list) and len(v)>3 and all(isinstance(x,(int,float)) for x in v):
            return f'n={len(v)}; min={min(v):.10g}; max={max(v):.10g}'
        if isinstance(v,dict) and set(v)=={'shape','min','max'}:
            return f"shape={v['shape']}; min={v['min']:.10g}; max={v['max']:.10g}"
        return json.dumps(v, ensure_ascii=False, separators=(',',':')) if isinstance(v,(dict,list)) else str(v)

    populations = snap['parameters']['populations']
    def parameter_table(name, params):
        story.append(p(name, h2))
        rows = [['参数标识（SI，除原历史核等）', '实际值 / 原数组范围']]
        rows.extend([k, value(v)] for k,v in params.items())
        story.extend([table(rows),Spacer(1,10)])

    def grouped_parameters(name, names):
        first = populations[names[0]]
        common = {k:v for k,v in first.items()
                  if all(k in populations[n] and populations[n][k] == v for n in names)}
        parameter_table(name + '：公共参数', common)
        story.append(p('适用类型：' + '；'.join(names), small))
        differences = []
        for n in names:
            for k,v in populations[n].items():
                if k not in common:
                    differences.append([n+' / '+k,value(v)])
        if differences:
            story.extend([table([['类型 / 独立参数', '实际值或范围']] + differences), Spacer(1,10)])

    for pop in ('rods','rod_BC','AII_AC'):
        parameter_table(pop, populations[pop])
    grouped_parameters('五类 Cone BC', ['midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC'])
    parameter_table('S-ON 保留支路（含继承元数据）',populations['local_feedforward_AC'])
    grouped_parameters('五类 RGC', ['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'])
    grouped_parameters('四类普通 AC', [n for n in populations if n.startswith(('same_','cross_'))])
    for name in ('route_gains','rod_phototransduction','TRPM1'):
        story.append(p(name, h2))
        story.extend([table([['参数/路由','当前值']] + [[k,value(v)] for k,v in snap['parameters'][name].items()]),Spacer(1,10)])
    story.append(p('H2 实际绑定', h2))
    story.append(table([['参数','值']] + [[k,value(v)] for k,v in snap['H2'].items()]))
    story.extend([PageBreak(),p('附录 B. 来源定位与复现身份', h1),
        p('当前模型 identity：'+snap['model_identity'],small),
        p('当前输入 source identity：'+snap['source_identity'],small),
        p('源 seed 随机种子：'+value(snap['rng_seed']),small),
        p('每条下列路径在生成快照时实际存在。完整 SHA256 保存在同目录“当前建模参数与来源.json”；此处保留路径和前 16 位供定位。哈希记录文档快照身份，不重新授予科学资格。')])
    for row in snap['source_files']:
        story.append(p(row['path']+'<HASH>'+row['sha256'][:16],small))
    # The marker separates a long path from its hash without building a wide table.
    for flow in story:
        if isinstance(flow,Paragraph) and '&lt;HASH&gt;' in flow.text:
            flow.__init__(flow.text.replace('&lt;HASH&gt;','<br/><font color="#708391">SHA256: ')+ '</font>', small)
    doc.multiBuild(story)
    print(json.dumps(dict(pdf=str(PDF), bytes=PDF.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-snapshot', action='store_true')
    args = parser.parse_args()
    if args.refresh_snapshot:
        refresh_snapshot()
    build()
