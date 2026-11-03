"""Reproducible LFP packet audit and exploratory direction decoding (run 009)."""
from __future__ import annotations
import base64
import csv
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_analysis_gui.core.xdf_io import load_streams

SOURCE = Path(r'E:\AitBCI\data\keaiduo\20261111\run-009-CO8-offline-LFP')
OUT = ROOT / 'output' / 'lfp_20261111_run009'
OUT.mkdir(parents=True, exist_ok=True)

def varint(b, i=0):
    value = 0
    for shift in range(0, 70, 7):
        c = b[i]; i += 1
        value |= (c & 127) << shift
        if not c & 128:
            return value, i
    raise ValueError('Invalid varint')

def fields(b):
    out = {}; i = 0
    while i < len(b):
        tag, i = varint(b, i)
        n, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = varint(b, i)
        elif wire in (1, 2, 5):
            size, i = varint(b, i) if wire == 2 else ((8 if wire == 1 else 4), i)
            value = b[i:i+size]; i += size
            assert len(value) == size
        else:
            raise ValueError(f'Unsupported protobuf wire {wire}')
        assert n not in out, 'Repeated field needs explicit handling'
        out[n] = value
    return out

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')

def main():
    streams, audit = load_streams(next((SOURCE / 'xdf').glob('*.xdf')))
    lfp = streams['lfp_packets']
    packets = [json.loads(row[0]) for row in lfp['time_series']]
    fs = [fields(base64.b64decode(p['payload_base64'], validate=True)) for p in packets]
    print('First protobuf fields:', {n: (f'bytes[{len(v)}] {v[:40]!r}' if isinstance(v,bytes) else v) for n,v in fs[0].items()}, flush=True)
    for key in ['sequence','sdk_sequence','packet_index','device_timestamp_100us','host_monotonic_ns','received_lsl_timestamp']:
        a=np.array([p[key] for p in packets]); d=np.diff(a)
        print(key, 'range',a[[0,-1]], 'step percentiles', np.percentile(d,[0,1,25,50,75,99,100]), 'nonpositive',sum(d<=0), 'unique',len(np.unique(a)),flush=True)
    for key in ['schema_version','mode','sample_format','samples_per_channel','wire_channel_ids','channel_mapping_generation','sdk_dropped_batches','pipe_dropped_frames']:
        print(key,Counter(str(p[key]) for p in packets).most_common(8),flush=True)
    dump('packet_audit_initial.json',dict(stream_audit=audit,first_packet=packets[0],last_packet=packets[-1]))
    analyze(streams, packets, fs, audit)

def analyze(streams, packets, proto, audit):
    ids=packets[0]['wire_channel_ids']; n=len(packets)
    # Cross-check protobuf metadata against the independent JSON wrapper on EVERY packet.
    for p,f in zip(packets,proto):
        for key,tag in [('schema_version',1),('mode',2),('sequence',5),('host_monotonic_ns',6),('device_timestamp_100us',7),('sample_format',9),('samples_per_channel',11),('sdk_sequence',12),('channel_mapping_generation',17)]:
            assert p[key]==f[tag], (key,p[key],f[tag])
        packed=[]; i=0
        while i<len(f[8]):
            v,i=varint(f[8],i); packed.append(v)
        assert packed==ids==p['wire_channel_ids']
        assert len(f[10])==p['sample_bytes']==6*20*2
    raw=np.stack([np.frombuffer(f[10],dtype='<i2') for f in proto])
    # Layout is inferred, not established without the acquisition .proto/SDK.
    x=raw.reshape(n,6,20).transpose(0,2,1).reshape(-1,6).astype(float)
    alternative=raw.reshape(-1,6).astype(float)
    layout={}
    for name,a in [('channel_major',x),('sample_major',alternative)]:
        layout[name]=dict(median_absolute_step=np.median(np.abs(np.diff(a,axis=0)),axis=0),
                          rms_step=np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=0)),channel_means=a.mean(axis=0),
                          lag1_correlation=[np.corrcoef(a[:-1,c],a[1:,c])[0,1] for c in range(6)])
    # Packet sequence is complete. Fit its rate to robust median host-time anchors.
    host=np.array([p['received_perf_counter_ns']/1e9 for p in packets])
    anchors=np.array([np.median(host[i:i+250]) for i in range(0,n-249,250)])
    seqanchors=np.array([np.median(np.arange(i,i+250)) for i in range(0,n-249,250)])
    slope,intercept=np.polyfit(seqanchors,anchors,1)
    residual=host-(intercept+slope*np.arange(n))
    # Low arrival-delay envelope; no claim of absolute acquisition-time calibration.
    intercept+=np.percentile(residual,5)
    rate=20/slope
    times=intercept+slope*np.arange(n)[:,None]-(19-np.arange(20))[None,:]/rate
    times=times.ravel()
    print('Layout',layout,'rate',rate,flush=True)
    np.savez_compressed(OUT/'unpacked_lfp.npz',values=x.astype(np.int16),sample_times_lsl_estimated=times,
                        channel_ids=ids,effective_sample_rate_hz=rate,packet_receive_times_lsl=streams['lfp_packets']['time_stamps'],
                        device_ticks=np.array([p['device_timestamp_100us'] for p in packets]))
    # Reuse event-defined trial reconstruction, without the spike-only loader.
    from data_analysis_gui.core.session import Session, reconstruct_trials
    from data_analysis_gui.core.catalog import describe
    path=next((SOURCE/'xdf').glob('*.xdf'))
    cursor=streams['cursortasks_cursor_state']; events=streams['cursortasks_events']
    session=Session(recording=describe(path,path.parent),counts=np.empty((0,0)),neural_times=times,metadata=np.empty((0,0)),metadata_labels=[],
        cursor_times=cursor['time_stamps'],cursor=cursor['time_series'],event_times=events['time_stamps'],events=events['time_series'],electrode_ids=np.array(ids))
    rows=reconstruct_trials(session)
    with (OUT/'reconstructed_trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); writer.writeheader(); writer.writerows(rows)
    # Eligible success trials: entire feature window before acquisition, fixed duration.
    eligible=[r for r in rows if r['complete'] and not r['incomplete_reason'] and r['outcome']=='success' and r['direction']>=0
              and r['acquire_time']>=.8 and r['start']-.65>=times[0] and r['start']+.85<=times[-1]]
    behavior=list(csv.DictReader((SOURCE/'behavior'/'trials.csv').open(encoding='utf-8-sig')))
    behavior={int(r['trial_num_overall']):r for r in behavior}
    for r in eligible:
        b=behavior[r['trial_id']]
        assert int(b['block_num'])==r['block'] and b['outcome']=='acquired'
        angle=np.degrees(np.arctan2(float(b['target_y']),float(b['target_x'])))%360
        assert int(round(angle/45))%8==r['direction']
    eligible_ids={r['trial_id'] for r in eligible}
    for r in rows:
        r['included_for_decoding']=r['trial_id'] in eligible_ids
    with (OUT/'reconstructed_trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); writer.writeheader(); writer.writerows(rows)
    print('Trials',len(rows),'outcomes',Counter(r['outcome'] for r in rows),'eligible',len(eligible),Counter(r['direction'] for r in eligible),flush=True)
    bands=[(4,8),(8,13),(13,30),(30,45),(65,95),(105,180)]
    windows={'early':(.1,.7),'late':(.2,.8),'pre':(-.6,0)}
    feats={}; psds=[]
    def segment_at(r,a,b,offset=0):
        i,j=np.searchsorted(times,[r['start']+a+offset,r['start']+b+offset])
        return x[i:j]
    for name,(a,b) in windows.items():
        bp=[]; wave=[]
        for r in eligible:
            z=segment_at(r,a,b)
            freq,power=signal.welch(z,fs=rate,nperseg=256,noverlap=128,axis=0,detrend='linear')
            bp.append(np.array([np.log10(np.maximum(power[(freq>=lo)&(freq<hi)].mean(axis=0),1e-12)) for lo,hi in bands]).T.ravel())
            # Six 100-ms mean-voltage bins; within-trial centering removes DC offsets.
            bins=np.array([segment_at(r,a+k*.1,a+(k+1)*.1).mean(axis=0) for k in range(6)])
            bins-=bins.mean(axis=0)
            wave.append(bins.T.ravel())
            if name=='early': psds.append(power)
        feats[name+'_power']=np.array(bp); feats[name+'_wave']=np.array(wave)
    feats['early_combined']=np.c_[feats['early_power'],feats['early_wave']]
    y=np.array([r['direction'] for r in eligible]); groups=np.array([r['block'] for r in eligible])
    folds=list(GroupKFold(5).split(y,y,groups))
    def predict(z,labels,foldlist=folds):
        pred=np.full(len(labels),-1)
        for tr,te in foldlist:
            model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage=.5))
            pred[te]=model.fit(z[tr],labels[tr]).predict(z[te])
        ok=pred>=0
        return float(balanced_accuracy_score(labels[ok],pred[ok])),pred
    with threadpool_limits(limits=1):
        results={}
        for name,z in feats.items():
            acc,pred=predict(z,y)
            results[name]=dict(balanced_accuracy=acc,accuracy=float(np.mean(pred==y)),confusion_matrix=confusion_matrix(y,pred,labels=np.arange(8)),predictions=pred)
        z=feats['early_power']; primary=results['early_power']['balanced_accuracy']
        rng=np.random.default_rng(20261111); null=[]
        for perm in range(499):
            yp=y.copy()
            for g in np.unique(groups):
                idx=np.flatnonzero(groups==g); yp[idx]=rng.permutation(yp[idx])
            null.append(predict(z,yp)[0])
            if perm%100==0: print('Permutation',perm,flush=True)
        primary_p=(1+sum(v>=primary for v in null))/(len(null)+1)
        singles=[predict(z[:,c*6:(c+1)*6],y)[0] for c in range(6)]
        bandpred=[predict(z[:,b::6],y) for b in range(6)]
        bandacc=[a for a,_ in bandpred]
        # Account for choosing the largest of six frequency-band results.
        rng=np.random.default_rng(20261112); max_band_null=[]
        for perm in range(499):
            yp=y.copy()
            for g in np.unique(groups):
                idx=np.flatnonzero(groups==g); yp[idx]=rng.permutation(yp[idx])
            max_band_null.append(max(predict(z[:,b::6],yp)[0] for b in range(6)))
        band_p=(1+sum(v>=max(bandacc) for v in max_band_null))/500
        # Band choice performed only inside each outer training fold.
        nested_pred=np.full(len(y),-1); chosen=[]
        for tr,te in folds:
            inner=list(GroupKFold(4).split(y[tr],y[tr],groups[tr]))
            inner_acc=[predict(z[tr,b::6],y[tr],inner)[0] for b in range(6)]
            best=int(np.argmax(inner_acc)); chosen.append(best)
            _,pred=predict(z[:,best::6],y,[(tr,te)]); nested_pred[te]=pred[te]
        nested_acc=float(balanced_accuracy_score(y,nested_pred))
        forward=[]
        for cutoff in (10,15,20):
            tr=np.flatnonzero(groups<=cutoff); te=np.flatnonzero((groups>cutoff)&(groups<=cutoff+5))
            if len(tr) and len(te):
                acc,pred=predict(z,y,[(tr,te)])
                inner=list(GroupKFold(4).split(y[tr],y[tr],groups[tr]))
                inner_acc=[predict(z[tr,b::6],y[tr],inner)[0] for b in range(6)]
                best=int(np.argmax(inner_acc)); acc_nested,_=predict(z[:,best::6],y,[(tr,te)])
                forward.append(dict(train_through_block=cutoff,test_blocks=[cutoff+1,cutoff+5],n_test=len(te),balanced_accuracy=acc,training_selected_band=best,training_selected_band_accuracy=acc_nested))
        sensitivity={}
        for offset in (-.02,.02):
            bp=[]
            for r in eligible:
                f,p=signal.welch(segment_at(r,.1,.7,offset),fs=rate,nperseg=256,noverlap=128,axis=0,detrend='linear')
                bp.append(np.array([np.log10(np.maximum(p[(f>=lo)&(f<hi)].mean(axis=0),1e-12)) for lo,hi in bands]).T.ravel())
            sensitivity[str(offset)]=predict(np.array(bp),y)[0]
    quality=dict(packet_count=n,sample_count=len(x),channel_ids=ids,inferred_layout=layout,effective_sample_rate_hz=rate,
        duration_seconds=float(times[-1]-times[0]),nominal_rate_hz=1000,sequence_gaps=int(np.sum(np.diff([p['sequence'] for p in packets])!=1)),
        pipe_dropped_frames_constant=666,raw_std=x.std(axis=0),raw_range=np.array([x.min(axis=0),x.max(axis=0)]),
        clipping_counts=np.sum((x==-32768)|(x==32767),axis=0),zero_fraction=np.mean(x==0,axis=0),correlation=np.corrcoef(x.T),
        receive_residual_percentiles_ms=np.percentile(residual,[0,5,50,95,100])*1000,
        device_ticks_per_second=float((packets[-1]['device_timestamp_100us']-packets[0]['device_timestamp_100us'])/(host[-1]-host[0])),
        reconstructed_trial_count=len(rows),eligible_count=len(y),direction_counts=np.bincount(y,minlength=8),
        trial_outcomes=dict(Counter(r['outcome'] for r in rows)),stream_audit=audit)
    summary=dict(quality=quality,primary='early_power',window_seconds=[.1,.7],bands_hz=bands,decoder='StandardScaler + LDA fixed shrinkage 0.5',
        cross_validation='5-fold GroupKFold by block; all features of one trial stay together',results=results,
        permutation=dict(count=len(null),within_block=True,p_value=primary_p,null_mean=float(np.mean(null)),null_95=np.percentile(null,[2.5,97.5]),null_scores=null),
        single_channel_accuracy=singles,band_accuracy=bandacc,
        band_comparison=dict(max_statistic_p_value=band_p,permutation_count=499,search_family=bands,max_band_null=max_band_null),
        nested_band_selection=dict(balanced_accuracy=nested_acc,selected_band_indices=chosen,predictions=nested_pred,confusion_matrix=confusion_matrix(y,nested_pred,labels=np.arange(8))),
        forward=forward,alignment_offset_sensitivity=sensitivity)
    # Descriptive block-bootstrap intervals for already-held-out predictions.
    rng=np.random.default_rng(1111); unique=np.unique(groups); intervals={}
    for key,pred in [('early_power',results['early_power']['predictions']),('nested_band_selection',nested_pred)]:
        values=[]
        for _ in range(2000):
            idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
            values.append(balanced_accuracy_score(y[idx],pred[idx]))
        intervals[key]=np.percentile(values,[2.5,97.5])
    summary['descriptive_block_bootstrap_95_intervals']=intervals
    summary['limitations']=[
        'No acquisition .proto or SDK available: int16 little-endian and channel-major layout inferred from payload size and channel-specific continuity/DC structure; voltage gain and channel-id physical mapping unverified.',
        'Estimated rate from sequence/host timestamps; absolute acquisition-to-task latency unknown. Device timestamp field name says 100 us but observed units track approximately 1 ms.',
        'Unusually weak low-frequency content and negative lag-1 autocorrelations: acquisition filtering or signal origin cannot be established from this recording alone.',
        'One manual joystick session, target direction labels: cannot distinguish motor intention, sensory cue responses, movement feedback or muscle/motion artifacts.',
        'Pre-target interval can contain previous-trial activity because ITI is zero; it is a temporal negative control, not a resting baseline.',
        'Only frequency-band search is corrected by the max statistic; model/window/channel exploration is exploratory. Bootstrap intervals condition on held-out predictions, not full training uncertainty.',
        'Trial-wise Welch detrending and waveform centering use the full fixed 600 ms window; no online/causal velocity-decoding claim.']
    summary['protobuf_wire_reference']='https://protobuf.dev/programming-guides/encoding/'
    dump('analysis_results.json',summary)
    np.savez_compressed(OUT/'decoding_features.npz',labels=y,blocks=groups,trial_ids=[r['trial_id'] for r in eligible],**feats)
    plots(x,times,rate,ids,quality,summary,psds,bands)
    print('Nested band selection',nested_acc,'selected',chosen,'six-band corrected p',band_p,'forward',forward,'intervals',intervals,flush=True)
    print('RESULTS', {k:round(v['balanced_accuracy'],4) for k,v in results.items()},'primary p',primary_p,flush=True)

def plots(x,times,rate,ids,quality,summary,psds,bands):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for c,ax in enumerate(axs.flat):
        s=int(rate*30); e=s+int(rate*2)
        ax.plot(times[s:e]-times[s],x[s:e,c],lw=.6)
        ax.set(title=f'Wire channel {ids[c]}',xlabel='Time (s)',ylabel='Raw ADC units (uncalibrated)')
    fig.savefig(OUT/'raw_signals.png',dpi=160); plt.close(fig)
    f,p=signal.welch(x,fs=rate,nperseg=2048,axis=0)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for c in range(6): axs[0].semilogy(f,p[:,c],label=str(ids[c]),lw=.9)
    axs[0].set(xlim=(1,200),xlabel='Frequency (Hz; estimated sample clock)',ylabel='PSD (raw units squared / Hz)',title='Continuous LFP spectrum'); axs[0].legend(ncol=3)
    im=axs[1].imshow(quality['correlation'],vmin=-1,vmax=1,cmap='RdBu_r'); axs[1].set(xticks=range(6),xticklabels=ids,yticks=range(6),yticklabels=ids,title='Raw channel correlation')
    fig.colorbar(im,ax=axs[1]); fig.savefig(OUT/'signal_quality.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True); axs=axes.ravel()
    names=['early_power','early_wave','early_combined','pre_power']
    vals=[summary['results'][k]['balanced_accuracy']*100 for k in names]
    vals.insert(3,summary['nested_band_selection']['balanced_accuracy']*100)
    axs[0].bar(range(5),vals,color=['#3979b7']*3+['#29836e','#999999']); axs[0].axhline(12.5,color='k',ls='--',label='Chance 12.5%')
    axs[0].set(xticks=range(5),xticklabels=['All bands','Waveform','Combined','Nested\nband choice','Pre-target\npower'],ylabel='Balanced accuracy (%)',ylim=(0,40),title='8 directions | 188 trials | 5 held-out-block folds')
    for i,v in enumerate(vals): axs[0].text(i,v+.5,f'{v:.1f}',ha='center')
    axs[0].legend(loc='upper right')
    cm=np.array(summary['nested_band_selection']['confusion_matrix']); row=cm/cm.sum(axis=1,keepdims=True)
    im=axs[1].imshow(row,vmin=0,vmax=max(.4,row.max()),cmap='Blues')
    for i in range(8):
        for j in range(8): axs[1].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
    axs[1].set(xticks=range(8),xticklabels=np.arange(8)*45,yticks=range(8),yticklabels=np.arange(8)*45,xlabel='Predicted direction (deg)',ylabel='Target direction (deg)',title='Nested band choice: held-out trial counts')
    null=np.array(summary['band_comparison']['max_band_null'])*100
    axs[2].hist(null,bins=20,color='#aaa'); axs[2].axvline(vals[3],color='#b84530',label=f'Observed {vals[3]:.1f}%')
    axs[2].set(xlabel='Best of 6 bands: balanced accuracy (%)',ylabel='Permutations',title=f'Within-block shuffle | six-band corrected p={summary["band_comparison"]["max_statistic_p_value"]:.3f}'); axs[2].legend()
    forward=summary['forward']; a=np.arange(len(forward))
    axs[3].bar(a-.18,[v['balanced_accuracy']*100 for v in forward],width=.36,label='All bands',color='#3979b7')
    axs[3].bar(a+.18,[v['training_selected_band_accuracy']*100 for v in forward],width=.36,label='Training-only band choice',color='#29836e')
    axs[3].axhline(12.5,color='k',ls='--'); axs[3].set(xticks=a,xticklabels=['Test blocks 11–15','Test blocks 16–20','Test blocks 21–25'],ylabel='Balanced accuracy (%)',ylim=(0,45),title='Forward prediction | train on earlier blocks only'); axs[3].legend()
    fig.savefig(OUT/'direction_decoding.png',dpi=180); plt.close(fig)

if __name__=='__main__':
    main()
