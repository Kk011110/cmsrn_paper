"""Train-fitted, input-history-only scenario labels and shared-metric summaries."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(ROOT))
from ablation_experiments.joint_full_v3.scripts.data import channel_metric

SOURCE = PACKAGE.parent / 'timeseries_all_models_curated_20261015'
EXPERT = PACKAGE.parent / 'expert_weights_contributions_20261015'
RUN = 'end_to_end_full_batch64_v2__seed42'
CITIES = ['chengdu', 'shanghai']
FLOWS = ['inflow', 'outflow']
LABELS = ['Low', 'Moderate', 'High', 'Extreme']
POLICY = dict(
    version='scenario_analysis_v1', seed=42, model_display_name='CMSRN', model_source_name='Full',
    fit_scope='Unique history indices of frozen training windows with source_quality=True; no validation/test fitting.',
    threshold_scope='Training windows whose 12 history slots are all source-available; all nodes; separate city and channel.',
    calendar='Actual Asia/Shanghai timestamps; weekday/weekend and 288 five-minute slots; never CSV split labels or model internal calendar.',
    reference='Median per day-type/time-slot/node/flow; if fewer than 4 observations pool day-types at same slot, then node median.',
    scale='max(1 count, 1.4826 * MAD of training residual around its median), per node/flow.',
    deviation='D = mean(abs((history-reference)/scale)) over 12 input steps.',
    variation='U = mean(abs(diff((history-reference)/scale))) over 11 differences.',
    background='B = mean over valid frozen CMSRN candidate sources of abs(mean_time(r_target)-mean_time(r_source)). No learned attention weights.',
    deviation_quantiles=[.50,.80,.95], ties='Use <= upper edge; duplicate edges create an empty interval, reported explicitly, never retune using test performance.',
    rapid_threshold='Training U Q75 among training D > training D Q80; Rapid if strictly greater.',
    background_thresholds='Training B Q(1/3,2/3) among training D > training D Q80.',
    high_deviation='D > training Q80', primary_horizon=6, horizons=list(range(1,13)),
    evaluation='Shared project channel_metric, float64, target > 0, no clipping, no extra removal of all-city-zero slots.',
    mechanism='Frozen-output removal, not retrained ablation; gates and other outputs unchanged.',
    uncertainty='Descriptive complete-test statistics; no IID pointwise confidence intervals or significance claims.',
    model_selection='Original locked joint-validation checkpoint, unchanged; no scenario-specific selection.',
)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def save_csv(path, rows):
    rows = list(rows)
    if not rows: raise ValueError(f'Empty output: {path}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def metric(pred, truth, mask):
    """Reuse exact main-project metric, with one selected flow duplicated for its BNT2 API."""
    mask = mask & (truth > 0)
    n = int(mask.sum())
    if not n: return n, None, None
    p = torch.from_numpy(np.repeat(np.asarray(pred[mask], np.float64)[:,None], 2, 1)).reshape(-1,1,1,2)
    y = torch.from_numpy(np.repeat(np.asarray(truth[mask], np.float64)[:,None], 2, 1)).reshape(-1,1,1,2)
    return n, float(channel_metric(p,y,'mae')[0]), float(channel_metric(p,y,'rmse')[0])

def scores(residual, histories, sources, valid):
    n, nodes = len(histories), residual.shape[1]
    d, u, bg = (np.empty((n,nodes,2), np.float32) for _ in range(3))
    for lo in range(0,n,64):
        hi=min(lo+64,n); r=residual[histories[lo:hi]]
        d[lo:hi]=np.abs(r).mean(1)
        u[lo:hi]=np.abs(np.diff(r,axis=1)).mean(1)
        b=r.mean(1)
        diff=np.abs(b[:,:,None,:]-b[:,sources,:])
        bg[lo:hi]=(diff*valid[None,:,:,None]).sum(2)/valid.sum(1)[None,:,None]
    return d,u,bg

def fit_city(city):
    print(city, 'fit history-only labels', flush=True)
    run=ROOT/'baselines/Full/outputs'/city/RUN
    cfg=read(run/'config_final.json'); root=Path(cfg['data_root'])
    split=ROOT/cfg['split']['path']
    assert sha(split)==cfg['split']['sha256']
    for name,digest in cfg['data_fingerprints'].items():
        assert sha(root/name)==digest, name
    with np.load(split) as z: h,y,tr,te=[z[k] for k in ('history','target','train','test')]
    raw=np.load(root/'all_data.npy',mmap_mode='r')
    quality=np.load(root/'source_quality_mask.npy').astype(bool)
    with (root/'timestamp_index.csv').open() as f: timestamps=np.array([r['timestamp'] for r in csv.DictReader(f)])
    dt=[datetime.fromisoformat(s) for s in timestamps]
    assert all((b-a).total_seconds()==300 for a,b in zip(dt,dt[1:]))
    daytype=np.array([x.weekday()>=5 for x in dt],int)
    slot=np.array([x.hour*12+x.minute//5 for x in dt],int)
    unique=np.unique(h[tr]); fit=unique[quality[unique]]
    valid_tr=tr[quality[h[tr]].all(1)]
    assert quality[h[te]].all() and quality[y[te]].all()
    global_median=np.median(raw[fit],axis=0)
    baseline=np.empty((2,288,1024,2),np.float32)
    support=np.zeros((2,288),int); fallback=np.zeros((2,288),int)
    for d in range(2):
        for s in range(288):
            ix=fit[(daytype[fit]==d)&(slot[fit]==s)]; support[d,s]=len(ix)
            if len(ix)<4: ix=fit[slot[fit]==s]; fallback[d,s]=1
            if not len(ix): baseline[d,s]=global_median;fallback[d,s]=2
            else: baseline[d,s]=np.median(raw[ix],axis=0)
    unscaled=np.asarray(raw,dtype=np.float32)-baseline[daytype,slot]
    center=np.median(unscaled[fit],axis=0)
    scale=np.maximum(1.,1.4826*np.median(np.abs(unscaled[fit]-center),axis=0)).astype(np.float32)
    residual=unscaled/scale
    with np.load(run/'artifacts/candidates.npz') as z: sources,valid=z['source_index'],z['neighbor_mask']
    assert (valid.sum(1)>0).all()
    td,tu,tb=scores(residual,h[valid_tr],sources,valid)
    dd,uu,bb=scores(residual,h[te],sources,valid)
    thresholds=[]; gids=np.empty(dd.shape,np.int8); rapid=np.empty(dd.shape,bool); bgids=np.empty(dd.shape,np.int8)
    for c,flow in enumerate(FLOWS):
        q=np.quantile(td[:,:,c], [.5,.8,.95])
        high=td[:,:,c]>q[1]
        assert high.any()
        uq=float(np.quantile(tu[:,:,c][high],.75)); bq=np.quantile(tb[:,:,c][high],[1/3,2/3])
        gids[:,:,c]=np.searchsorted(q,dd[:,:,c],side='left')
        rapid[:,:,c]=uu[:,:,c]>uq
        bgids[:,:,c]=np.searchsorted(bq,bb[:,:,c],side='left')
        thresholds.append(dict(flow=flow,deviation_edges=q.tolist(),rapid_u_q75=uq,background_edges=bq.tolist(),
                               tied_deviation_edges=bool(len(set(q))<3),tied_background_edges=bool(bq[0]==bq[1]),
                               train_high_deviation_points=int(high.sum()),scale_at_floor_nodes=int((scale[:,c]==1).sum()),
                               train_group_counts=np.bincount(np.searchsorted(q,td[:,:,c].reshape(-1),side='left'),minlength=4).tolist()))
    citydir=PACKAGE/'data'/city;citydir.mkdir(exist_ok=True)
    np.savez_compressed(citydir/'train_reference.npz',baseline=baseline,scale=scale,fit_time_indices=fit,
                        train_window_rows=valid_tr,support=support,fallback=fallback,source_index=sources,neighbor_mask=valid)
    np.savez_compressed(citydir/'test_groups.npz',deviation=dd,variation=uu,background_difference=bb,
                        deviation_group=gids,rapid=rapid,background_group=bgids,history_indices=h[te],target_indices=y[te],
                        test_window_rows=te,forecast_origin_timestamps=timestamps[h[te,-1]],target_h6_timestamps=timestamps[y[te,5]],
                        history_mean_raw=np.asarray(raw[h[te]]).mean(1),channel_names=np.array(FLOWS))
    provenance=dict(city=city,train_history_unique=len(unique),fit_unique_available=len(fit),training_windows=len(tr),
                    threshold_training_windows=len(valid_tr),test_windows=len(te),fit_first_timestamp=str(timestamps[fit[0]]),
                    fit_last_timestamp=str(timestamps[fit[-1]]),test_start=str(timestamps[y[te[0],5]]),test_end=str(timestamps[y[te[-1],5]]),
                    thresholds=thresholds,calendar_type_counts={str(d):int((daytype[fit]==d).sum()) for d in [0,1]},
                    sources={str(p.relative_to(ROOT)):sha(p) for p in [split,root/'timestamp_index.csv',root/'source_quality_mask.npy',
                             root/'all_data.npy',run/'artifacts/candidates.npz',run/'config_final.json']},
                    raw_data_path=str(root/'all_data.npy'),checkpoint=read(EXPERT/city/'metadata.json'),
                    calendar_warning='External reference uses actual timestamps; deliberately independent of internal S0 calendar.',
                    background_proxy='Raw standardized residual mean, not learned BCMS low-frequency features; fixed CMSRN train-fitted candidate set.')
    write(citydir/'thresholds_and_provenance.json',provenance)
    print(city, 'thresholds', thresholds, flush=True)
    return provenance

def performance_and_serr():
    methods=read(SOURCE/'methods.json')['methods']; perf=[]; serr=[]; coverage=[]; checks=[]; flow_control=[]
    for city in CITIES:
        with np.load(PACKAGE/'data'/city/'test_groups.npz') as z: groups={k:z[k] for k in z.files}
        with np.load(SOURCE/city/'truth_h6.npz') as z: truth={k:z[k] for k in z.files}
        y=truth['ground_truth_channels'].reshape(-1,1024,2).astype(np.float64)
        for key in ('history_indices','test_window_rows'): np.testing.assert_array_equal(groups[key],truth[key])
        np.testing.assert_array_equal(groups['target_indices'][:,5],truth['target_h6_indices'])
        with np.load(SOURCE/city/'predictions/Full_h6.npz') as z: full=z['prediction_channels'].reshape(y.shape).astype(np.float64)
        group=groups['deviation_group']; full_stats={}
        for c,flow in enumerate(FLOWS):
            for g in range(4):
                mask=group[:,:,c]==g; n,mae,rmse=metric(full[:,:,c],y[:,:,c],mask)
                full_stats[c,g]=(n,mae,rmse)
                eff=mask&(y[:,:,c]>0); origins=groups['forecast_origin_timestamps'][eff.any(1)]
                coverage.append(dict(city=city,flow=flow,group_id=g,group_label=LABELS[g],all_points=int(mask.sum()),n=n,
                                     n_windows=int(eff.any(1).sum()),n_nodes=int(eff.any(0).sum()),
                                     n_origin_dates=len(set(str(s)[:10] for s in origins)),
                                     mean_positive_target=float(y[:,:,c][eff].mean()) if n else None,
                                     mean_history_flow=float(groups['history_mean_raw'][:,:,c][eff].mean()) if n else None,
                                     full_mae=mae,full_rmse=rmse))
        # Model-independent flow-scale control: fixed training quantiles of history mean.
        cfg=read(SOURCE/city/'provenance/Full/config_final.json'); raw=np.load(Path(cfg['data_root'])/'all_data.npy',mmap_mode='r')
        with np.load(PACKAGE/'data'/city/'train_reference.npz') as z: tr=z['train_window_rows']
        with np.load(ROOT/cfg['split']['path']) as z: histories=z['history'][tr]
        train_mean=np.asarray(raw[histories]).mean(1)
        flow_edges=[np.quantile(train_mean[:,:,c],[.5,.8,.95]) for c in range(2)]
        flow_groups=np.stack([np.searchsorted(flow_edges[c],groups['history_mean_raw'][:,:,c],side='left') for c in range(2)],-1)
        write(PACKAGE/'data'/city/'flow_control_thresholds.json',dict(edges=[q.tolist() for q in flow_edges],scope='available training history windows; all nodes; mean raw 12-step history'))
        for method in methods:
            src=SOURCE/city/'predictions'/f'{method}_h6.npz'
            with np.load(src) as z:
                p=z['prediction_channels'].reshape(y.shape).astype(np.float64)
                for key in ('test_window_rows','target_h6_indices'):
                    if key in z: np.testing.assert_array_equal(z[key],truth[key])
            assert np.isfinite(p).all()
            ref=read(SOURCE/city/'provenance'/method/'metrics.json')
            for c,flow in enumerate(FLOWS):
                n_all,ma,rm=metric(p[:,:,c],y[:,:,c],np.ones(y.shape[:2],bool))
                for name,actual in [('mae',ma),('rmse',rm)]:
                    expected=ref['horizons'][flow][5][name]
                    passed=bool(np.isclose(actual,expected,rtol=1e-4,atol=1e-5))
                    checks.append(dict(city=city,model=method,flow=flow,metric=name,actual=actual,reference=expected,passed=passed))
                    assert passed, checks[-1]
                subgroup=[]
                for g in range(4):
                    n,mae,rmse=metric(p[:,:,c],y[:,:,c],group[:,:,c]==g)
                    full_mae=full_stats[c,g][1]
                    perf.append(dict(city=city,flow=flow,model=method,group_id=g,group_label=LABELS[g],n=n,mae=mae,rmse=rmse,
                                     full_mae=full_mae,improvement_pct=100*(mae-full_mae)/mae if n and mae else None))
                    subgroup.append((n,mae,rmse))
                    for f in range(4):
                        mask=(group[:,:,c]==g)&(flow_groups[:,:,c]==f)
                        nn,mm,rr=metric(p[:,:,c],y[:,:,c],mask)
                        flow_control.append(dict(city=city,flow=flow,model=method,deviation_group=g,history_flow_group=f,n=nn,mae=mm,rmse=rr))
                assert sum(t[0] for t in subgroup)==n_all
                np.testing.assert_allclose(sum(n*m for n,m,r in subgroup if n)/n_all,ma,rtol=1e-12)
                np.testing.assert_allclose(np.sqrt(sum(n*r*r for n,m,r in subgroup if n)/n_all),rm,rtol=1e-12)
            print(city,method,'H6 verified',flush=True)
        with np.load(EXPERT/city/'all_test_h6.npz') as z:
            ep=z['prediction_channels'].astype(np.float64); ey=z['ground_truth_channels'].astype(np.float64)
            cs=z['self_contribution'].astype(np.float64);cd=z['delay_contribution'].astype(np.float64); w=z['weights'].astype(np.float64)
        np.testing.assert_array_equal(ep,full);np.testing.assert_array_equal(ey,y)
        for c,flow in enumerate(FLOWS):
            for rapid,name in [(False,'Slow'),(True,'Rapid')]:
                mask=(group[:,:,c]>=2)&(groups['rapid'][:,:,c]==rapid)
                n,fma,frm=metric(full[:,:,c],y[:,:,c],mask); effective=mask&(y[:,:,c]>0)
                for branch,contribution,ww in [('Self',cs[:,:,c],w[:,:,c,0]),('Delay',cd[:,:,c],w[:,:,c,1]),('SERR',cs[:,:,c]+cd[:,:,c],None)]:
                    _,rma,rrm=metric(full[:,:,c]-contribution,y[:,:,c],mask)
                    serr.append(dict(city=city,flow=flow,subgroup=name,branch=branch,n=n,full_mae=fma,removed_mae=rma,
                                     delta_mae=rma-fma if n else None,delta_rmse=rrm-frm if n else None,
                                     mean_weight=float(ww[effective].mean()) if n and ww is not None else None,
                                     mean_abs_contribution=float(np.abs(contribution[effective]).mean()) if n else None))
    save_csv(PACKAGE/'data/performance_h6.csv',perf)
    save_csv(PACKAGE/'data/scenario_coverage.csv',coverage)
    save_csv(PACKAGE/'data/serr_h6.csv',serr)
    save_csv(PACKAGE/'data/history_flow_control.csv',flow_control)
    write(PACKAGE/'data/h6_validation.json',dict(passed=True,checks=checks,metric_checks=len(checks),
          weighted_group_reconstruction='MAE weighted by positive count; RMSE by weighted squared RMSE; all 68 model/channel pairs passed',
          expert_h6_identity='Bit-identical with original Full H6 and truth in both cities'))

def bcms():
    rows=[];checks=[]
    for city in CITIES:
        source=PACKAGE/'inference'/city/'all_horizons.npz'
        if not source.exists(): raise FileNotFoundError(source)
        with np.load(PACKAGE/'data'/city/'test_groups.npz') as z: groups={k:z[k] for k in z.files}
        with np.load(source) as z:
            p=z['prediction_channels'].astype(np.float64);cg=z['bgst_contribution'].astype(np.float64)
            for key in ('history_indices','target_indices','test_window_rows'):
                np.testing.assert_array_equal(z[key],groups[key])
        cfg=read(SOURCE/city/'provenance/Full/config_final.json')
        raw=np.load(Path(cfg['data_root'])/'all_data.npy',mmap_mode='r')
        y=np.asarray(raw[groups['target_indices']]).transpose(0,2,1,3).astype(np.float64)
        assert p.shape==cg.shape==y.shape
        with np.load(EXPERT/city/'all_test_h6.npz') as z:
            np.testing.assert_allclose(p[:,:,5,:],z['prediction_channels'],rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(cg[:,:,5,:],z['bgst_contribution'],rtol=1e-5,atol=1e-5)
        for c,flow in enumerate(FLOWS):
            for scope in ['high_deviation','all_test']:
                mask0=groups['deviation_group'][:,:,c]>=2 if scope=='high_deviation' else np.ones(y.shape[:2],bool)
                for g,name in enumerate(['Low','Medium','High']):
                    mask=mask0&(groups['background_group'][:,:,c]==g)
                    for h in range(12):
                        n,fma,frm=metric(p[:,:,h,c],y[:,:,h,c],mask)
                        _,rma,rrm=metric(p[:,:,h,c]-cg[:,:,h,c],y[:,:,h,c],mask)
                        rows.append(dict(city=city,flow=flow,scope=scope,background_group=name,horizon=h+1,n=n,full_mae=fma,
                                         removed_mae=rma,delta_mae=rma-fma if n else None,delta_rmse=rrm-frm if n else None))
        checks.append(dict(city=city,h6_prediction_and_contribution_matches=True,shape=list(p.shape)))
    save_csv(PACKAGE/'data/bcms_horizons.csv',rows)
    write(PACKAGE/'data/bcms_validation.json',dict(passed=True,checks=checks))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['groups','h6','bcms','all'],default='all')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage in ['groups','all']:
        write(PACKAGE/'POLICY.json',POLICY)
        for city in CITIES: fit_city(city)
    if args.stage in ['h6','all']: performance_and_serr()
    if args.stage in ['bcms','all']: bcms()
