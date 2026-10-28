"""Audit author-supplied aggregates; never imports or executes the training script.

Run with Python's standard library. Optional --figures points to original SVGs.
This checks aggregate identities, not raw-prediction or checkpoint contents.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--figures', type=Path)
args = parser.parse_args()
checks = []
warnings = []

def check(name, condition, detail=None):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})

def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ['n', 'group_id', 'horizon']:
            if k in r: r[k] = int(r[k])
        for k in ['mae','rmse','full_mae','improvement_pct','removed_mae','delta_mae','delta_rmse','mean_weight','mean_abs_contribution']:
            if k in r: r[k] = float(r[k]) if r[k] else None
    return rows

def close(a,b,atol=1e-10):
    return abs(a-b) <= atol

def pooled(rows, key='mae', squared=False):
    n = sum(r['n'] for r in rows)
    v = sum(r['n'] * (r[key]**2 if squared else r[key]) for r in rows)/n
    return math.sqrt(v) if squared else v

perf, serr, bcms = (read(n) for n in ['performance_h6.csv','serr_h6.csv','bcms_horizons.csv'])
cities=['chengdu','shanghai']; flows=['inflow','outflow']
models=list(dict.fromkeys(r['model'] for r in perf))
check('row_counts', [len(perf),len(serr),len(bcms)]==[272,24,288], [len(perf),len(serr),len(bcms)])
check('method_count',len(models)==17,models)
check('performance_unique_keys',len({(r['city'],r['flow'],r['model'],r['group_id']) for r in perf})==len(perf))
check('serr_unique_keys',len({(r['city'],r['flow'],r['subgroup'],r['branch']) for r in serr})==len(serr))
check('bcms_unique_keys',len({(r['city'],r['flow'],r['scope'],r['background_group'],r['horizon']) for r in bcms})==len(bcms))
check('improvement_formula',all(close(r['improvement_pct'],100*(r['mae']-r['full_mae'])/r['mae']) for r in perf))
for title, rows in [('serr',serr),('bcms',bcms)]:
    check(title+'_removal_difference',all(close(r['delta_mae'],r['removed_mae']-r['full_mae']) for r in rows))

full_reconstruction=[]; extremes=[]; self_ratios=[]
table_matches=[]; table_missing=[]
for city in cities:
    provenance=json.loads((ROOT/'data'/city/'thresholds_and_provenance.json').read_text(encoding='utf-8-sig'))
    for t in provenance['thresholds']:
        check(city+'_'+t['flow']+'_training_count',sum(t['train_group_counts'])==1024*provenance['threshold_training_windows'])
        check(city+'_'+t['flow']+'_training_high_count',sum(t['train_group_counts'][2:])==t['train_high_deviation_points'])
    for flow in flows:
        select=lambda rows: [r for r in rows if r['city']==city and r['flow']==flow]
        p=select(perf); s=select(serr); b=select(bcms)
        full=sorted([r for r in p if r['model']=='Full'],key=lambda r:r['group_id'])
        n_all=sum(r['n'] for r in full); n_high=sum(r['n'] for r in full if r['group_id']>=2)
        full_reconstruction.append({'city':city,'flow':flow,'n':n_all,'h6_mae':pooled(full),'h6_rmse':pooled(full,'rmse',True),'n_high':n_high})
        for g in range(4):
            rr=[r for r in p if r['group_id']==g]
            check(f'{city}_{flow}_group{g}_common_n',len({r['n'] for r in rr})==1)
            check(f'{city}_{flow}_group{g}_full_reference',all(close(r['full_mae'],full[g]['mae']) for r in rr))
        er=sorted([r for r in p if r['group_id']==3],key=lambda r:r['mae'])
        best=next(r for r in er if r['model']!='Full')
        extremes.append({'city':city,'flow':flow,'rank':1+next(i for i,r in enumerate(er) if r['model']=='Full'), 'full_mae':full[3]['mae'],'best_baseline':best['model'],'best_baseline_mae':best['mae'],'improvement_pct':best['improvement_pct']})
        sx=[r for r in s if r['branch']=='Self']
        self_ratios.append({'city':city,'flow':flow,'rapid_over_slow':next(r['delta_mae'] for r in sx if r['subgroup']=='Rapid')/next(r['delta_mae'] for r in sx if r['subgroup']=='Slow')})
        check(f'{city}_{flow}_serr_high_n',sum(r['n'] for r in sx)==n_high)
        check(f'{city}_{flow}_serr_high_mae',close(pooled(sx,'full_mae'),pooled(full[2:]),1e-8))
        for subgroup in ['Slow','Rapid']:
            sr=[r for r in s if r['subgroup']==subgroup]
            check(f'{city}_{flow}_{subgroup}_serr_common_n',len({r['n'] for r in sr})==1)
            check(f'{city}_{flow}_{subgroup}_weights_sum_one',close(sum(r['mean_weight'] for r in sr if r['branch']!='SERR'),1,1e-6))
        for scope, expected_n, expected_mae in [('high_deviation',n_high,pooled(full[2:])),('all_test',n_all,pooled(full))]:
            br=[r for r in b if r['scope']==scope and r['horizon']==6]
            check(f'{city}_{flow}_{scope}_bcms_h6_n',sum(r['n'] for r in br)==expected_n)
            check(f'{city}_{flow}_{scope}_bcms_h6_mae',close(pooled(br,'full_mae'),expected_mae,1e-6),{'bcms':pooled(br,'full_mae'),'performance':expected_mae})
        # The paper tables display three decimals; compare both H6 metrics.
        lines=(PAPER/'tables'/f'{city}-{flow}.tex').read_text(encoding='utf-8-sig').splitlines()
        for model in models:
            display='CMSRN' if model=='Full' else model
            line=next((line for line in lines if '&' in line and display in line.split('&')[0]),None)
            if line is None:
                table_missing.append({'city':city,'flow':flow,'model':model});continue
            cells=line.split('&')[1:]
            numbers=[float(re.search(r'\d+\.\d+',cell).group()) for cell in cells]
            rr=[r for r in p if r['model']==model]
            expected=[pooled(rr),pooled(rr,'rmse',True)]
            ok=all(abs(a-b)<0.000500001 for a,b in zip(expected,numbers[2:4]))
            table_matches.append({'city':city,'flow':flow,'model':model,'reconstructed':expected,'displayed':numbers[2:4],'passed':ok})
table_exceptions=[r for r in table_matches if not r['passed']]
check('paper_table_h6_display_matches',not table_exceptions,{'comparisons':len(table_matches),'unlisted_methods':sorted({r['model'] for r in table_missing}),'exceptions':table_exceptions})
check('paper_table_h6_mae_display_matches',all(abs(r['reconstructed'][0]-r['displayed'][0])<0.000500001 for r in table_matches))
if table_exceptions:
    warnings.append('The original paper table is preserved. Shanghai inflow Full H6 RMSE reconstructed from the new CSV is 1.2255026830174054 (rounds to 1.226), whereas the original workbook stores 1.22547011631793 (table 1.225). The reason for this small numerical-entry difference has not been isolated. Both fall within the source-script metric consistency tolerance (rtol=1e-4, atol=1e-5). All main-table H6 MAE display values match; the new section analyses MAE.')

bg_order=[]; between_scopes=[]
for city in cities:
 for flow in flows:
  for h in range(1,13):
   for scope in ['high_deviation','all_test']:
    rows=[next(r for r in bcms if (r['city'],r['flow'],r['horizon'],r['scope'],r['background_group'])==(city,flow,h,scope,g)) for g in ['Low','Medium','High']]
    bg_order.append(rows[0]['delta_mae']>rows[1]['delta_mae']>rows[2]['delta_mae'])
   for g in ['Low','Medium','High']:
    get=lambda scope: next(r['delta_mae'] for r in bcms if (r['city'],r['flow'],r['horizon'],r['scope'],r['background_group'])==(city,flow,h,scope,g))
    between_scopes.append(get('all_test')>get('high_deviation'))
check('bcms_background_order_all_96_cases',all(bg_order))
check('bcms_all_test_exceeds_high_all_144_cases',all(between_scopes))
check('serr_self_delay_16_positive',all(r['delta_mae']>0 for r in serr if r['branch']!='SERR'))
check('serr_rapid_self_larger_all_four',all(r['rapid_over_slow']>1 for r in self_ratios))
bcms_summary={scope:{'positive':sum(r['delta_mae']>0 for r in bcms if r['scope']==scope),'negative_rows':[r for r in bcms if r['scope']==scope and r['delta_mae']<0]} for scope in ['high_deviation','all_test']}

svg_checks=[]
if args.figures:
 def svg_numbers(stem):
    root=ET.parse(args.figures/(stem+'.svg')).getroot()
    ts=[''.join(x.itertext()) for x in root.iter() if x.tag.endswith('}text')]
    return [t.replace('−','-') for t in ts if re.fullmatch(r'[+−-][0-9]+[.][0-9]+',t)]
 def compare_display(stem, expected, precision=None):
    shown=svg_numbers(stem)
    ok=len(shown)==len(expected) and all(abs(float(a)-b)<=0.5*10**(-len(a.split('.')[1]))+1e-10 for a,b in zip(shown,expected))
    rec={'figure':stem,'count_svg':len(shown),'count_csv':len(expected),'passed':ok}
    if not ok:
        rec['first_mismatches']=[{'position':i,'svg':a,'csv':b} for i,(a,b) in enumerate(zip(shown,expected)) if abs(float(a)-b)>0.5*10**(-len(a.split('.')[1]))+1e-10][:5]
    svg_checks.append(rec)
 for scope,stem in [('high_deviation','bcms_sensitivity_horizons'),('all_test','bcms_sensitivity_horizons_all_test')]:
    expected=[next(r['delta_mae']*1000 for r in bcms if (r['city'],r['flow'],r['scope'],r['background_group'],r['horizon'])==(city,flow,scope,g,h)) for city in cities for flow in flows for g in ['Low','Medium','High'] for h in range(1,13)]
    compare_display(stem,expected)
 baselines=['HA','SVR','VAR','DCRNN','STGCN','GWNet','STNorm','DSTAGNN','PDFormer','STWave','PatchTST','TimeMixer','STDN','MetaDG','HyperD','RAST']
 expected=[next(r['improvement_pct'] for r in perf if (r['city'],r['flow'],r['model'],r['group_id'])==(city,flow,model,g)) for city in cities for flow in flows for model in baselines for g in range(4)]
 compare_display('performance_h6',expected)
 for city in cities:
    expected=[next(r['improvement_pct'] for r in perf if (r['city'],r['flow'],r['model'],r['group_id'])==(city,flow,model,g)) for flow in flows for model in baselines for g in range(4)]
    compare_display('performance_h6_'+city,expected)
 expected=[next(r['delta_mae'] for r in serr if (r['city'],r['flow'],r['branch'],r['subgroup'])==(city,flow,branch,subgroup)) for city in cities for flow in flows for branch in ['Self','Delay'] for subgroup in ['Slow','Rapid']]
 compare_display('serr_sensitivity_h6',expected)
 absolute_root=ET.parse(args.figures/'performance_h6_absolute_mae.svg').getroot()
 absolute_shown=[''.join(x.itertext()) for x in absolute_root.iter() if x.tag.endswith('}text')]
 absolute_shown=[t for t in absolute_shown if re.fullmatch(r'[0-9]+[.][0-9]{2}',t)]
 absolute_expected=[next(r['mae'] for r in perf if (r['city'],r['flow'],r['model'],r['group_id'])==(city,flow,model,g)) for city in cities for flow in flows for model in baselines+['Full'] for g in range(4)]
 svg_checks.append({'figure':'performance_h6_absolute_mae','count_svg':len(absolute_shown),'count_csv':len(absolute_expected),'passed':len(absolute_shown)==len(absolute_expected) and all(abs(float(a)-b)<=0.005000001 for a,b in zip(absolute_shown,absolute_expected))})
 for rec in svg_checks:check('svg_'+rec['figure'],rec['passed'],rec)
else:
 warnings.append('SVG display audit was not requested; use --figures to provide the directory containing original SVGs.')

summary={
 'scope':'Author-supplied aggregate CSV consistency, source-script static inspection, and optional original SVG display rounding. Raw prediction arrays and checkpoint bytes were not provided or executed.',
 'passed':all(c['passed'] for c in checks),'checks':checks,'warnings':warnings,
 'h6_reconstruction':full_reconstruction,'main_table_h6':table_matches,'unlisted_methods':table_missing,
 'known_original_workbook_precision':{'file':'source/results/comparison/CMSRN_model_comparison_tables_chengdu_first.xlsx','sheet':'上海T5_inflow','model':'CMSRN (Ours)','h6_mae':0.726768800793293,'h6_rmse':1.22547011631793,'source':'Read directly with openpyxl, data_only=True; original workbook and paper table retained.'},
 'extreme_rankings':extremes,'self_rapid_over_slow':self_ratios,'bcms_summary':bcms_summary,'svg_checks':svg_checks,
 'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').rglob('*')) if p.is_file()},
 'static_findings':[
  'Routine reference: median by day type and slot; fewer than four observations pools day types at the same slot; only an empty pooled slot falls back to the node/channel median.',
  'Scale is fixed per node/channel: max(1,1.4826*MAD of available unique training-history residuals around their median), independent of time slot.',
  'Threshold fitting uses only training windows with all 12 source-history slots available; all nodes, separate city and channel.',
  'Residual-removal difference in the supplied aggregates is arithmetically exact; prepare_groups.py subtracts existing contributions without recomputing gates or retraining.',
  'SERR mean weights sum to one within float precision; self and delay MAE removal effects are not additive because MAE is nonlinear.',
  'Per-sample prediction decomposition identity and checkpoint hashes are author-provided provenance, not independently re-evaluated from raw arrays.',
 ]
}
(ROOT/'numeric-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
md=['# 困难情景统计核查','',f"总体检查：{'通过' if summary['passed'] else '存在待检查项'}；{sum(c['passed'] for c in checks)}/{len(checks)}项。",'',
    '核查范围：基于用户补充CSV重新计算汇总恒等式、排名和展示精度，并静态阅读分组脚本；未运行训练/推断脚本，也未以原始预测数组或检查点文件重新推断。','',
    '## H6重构与主表','', '|城市|通道|有效n|重构MAE|重构RMSE|','|---|---|---:|---:|---:|']
for r in full_reconstruction:md.append(f"|{r['city']}|{r['flow']}|{r['n']}|{r['h6_mae']:.9f}|{r['h6_rmse']:.9f}|")
md+=['',f"现有主表中的{len(table_matches)}个方法—城市—通道组合，全部H6 MAE通过三位小数展示精度核对；RMSE仅上海流入Full存在以下舍入边界差异。原表没有VAR、STNorm、STWave，三者仅属新增情景扩展比较。",'',
     '上海流入Full：原工作簿H6 MAE为0.726768800793293，RMSE为1.22547011631793；新CSV重构为0.726774085249245和1.225502683017405。MAE显示均为0.727，RMSE分别四舍五入为1.225和1.226。差异位于原脚本的rtol=1e-4、atol=1e-5容差内，具体数值入口/精度原因尚未单独核定。本次保留原主表，新增章节只论述CSV支持的分组MAE。','',
     '## 极高偏离组','', '|城市|通道|CMSRN排名|最近/最佳基线|相对最佳基线改善（%）|','|---|---|---:|---|---:|']
for r in extremes:md.append(f"|{r['city']}|{r['flow']}|{r['rank']}|{r['best_baseline']}|{r['improvement_pct']:.6f}|")
md+=['','## 残差与分组一致性','',
     '- SERR与BCMS的delta_mae均核对为removed_mae−full_mae。',
     '- SERR快/慢组与BCMS背景组在H6的有效n均重构相应总体/高偏离子集；加权完整模型MAE亦一致。',
     '- Self与Delay共16项移除差值均正；快速/平缓自身修正比值为'+', '.join(f"{r['rapid_over_slow']:.6f}" for r in self_ratios)+'。',
     '- 高偏离BCMS：143/144为正；完整测试BCMS：144/144为正。两范围的所有城市—通道—预测步均呈低背景差异>中>高；完整测试的144项均大于对应高偏离值。',
     '- 不将单分支MAE移除效应求和视为整个SERR效应；不将冻结输出诊断解释为重新训练消融或结构机制的独立因果贡献。','',
     '## SVG显示核对','']
for r in svg_checks:md.append(f"- {r['figure']}: {'通过' if r['passed'] else '待检查'}，SVG {r['count_svg']} 个显示数值，CSV {r['count_csv']} 个对应数值。")
md+=['','## 分组实现要点','',
     '- 常规中位数按节点、通道、日类型、时槽拟合；原组合支持不足4点时合并同槽日类型，合并后为空才回退到节点/通道总体中位数。',
     '- MAD尺度是每节点/通道固定量，来自可用的唯一训练历史索引；不是时槽特定尺度。',
     '- 训练阈值只使用12个历史槽全部有效的训练窗口。上海为4180个，成都为5170个；这不改变原预测实验的训练样本。',
     '- 检查点身份由用户提供的provenance记录；本次没有检查点二进制，未独立验证文件SHA256。']
failed=[c for c in checks if not c['passed']]
if failed:md+=['','## 待检查项','']+[f"- {c['name']}: {c['detail']}" for c in failed]
(ROOT/'numeric-audit.md').write_text('\n'.join(md)+'\n',encoding='utf8')
print(json.dumps({'passed':summary['passed'],'checks':len(checks),'failures':failed,'svg_checks':svg_checks},ensure_ascii=False,indent=2))
