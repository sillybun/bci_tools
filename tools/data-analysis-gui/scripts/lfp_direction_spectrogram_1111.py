"""Direction-aligned LFP power spectrograms and block-aware response tests.

188 pre-reviewed successful trials, six channels, eight target directions.
The pre-event reference is previous-task activity, not physiological rest.
"""
import csv
import json
import hashlib
import shutil
from pathlib import Path
import sys
import numpy as np
from scipy import signal
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_analysis_gui.core.analysis import bh_fdr

BASE=ROOT/'output/lfp_20261111_run009'
OUT=BASE/'direction_spectrogram'
DELIVERY=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures/LFP_Direction_Spectrogram')
BANDS=[(4,8),(8,13),(13,30),(30,45),(65,95),(105,180)]
N_PERM=1999

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')

def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tf_data():
    raw=np.load(BASE/'unpacked_lfp.npz');x=raw['values'].astype(float);times=raw['sample_times_lsl_estimated'];fs=float(raw['effective_sample_rate_hz'])
    feature=np.load(BASE/'decoding_features.npz');ids=feature['trial_ids'];y=feature['labels'];blocks=feature['blocks']
    source=list(csv.DictReader((BASE/'reconstructed_trials.csv').open(encoding='utf-8-sig')))
    by_id={int(r['trial_id']):r for r in source if int(r['trial_id'])>=0}
    before=int(round(.85*fs));after=int(round(.95*fs));maps=[];power=[];post=[];pre=[];alignment=[]
    for tid,d,b in zip(ids,y,blocks):
        r=by_id[int(tid)];assert r['included_for_decoding']=='True' and int(r['direction'])==d and int(r['block'])==b
        start=float(r['start']);j=int(np.searchsorted(times,start));j=min((j-1,j),key=lambda a:abs(times[a]-start))
        assert j-before>=0 and j+after<=len(x)
        epoch=x[j-before:j+after]
        freq,t,psd=signal.spectrogram(epoch,fs=fs,window='hann',nperseg=256,noverlap=236,nfft=256,
                                     detrend='linear',scaling='density',mode='psd',axis=0)
        t=t-before/fs;alignment.append(times[j]-start)
        # Entire analysis windows remain before the current trial's acquisition.
        baseline=(t>=-.47)&(t<=-.23);active=(t>=.23)&(t<=.57)
        assert baseline.any() and active.any()
        half=128/fs
        assert t[active].max()+half < min(.7,float(r['acquire_time']))+1e-8
        assert t[baseline].max()+half < -.1+1e-8
        p=psd.transpose(1,0,2)
        logp=10*np.log10(np.maximum(p,1e-12))
        keepf=(freq>=4)&(freq<=200);keept=(t>=-.6)&(t<=.65)
        maps.append((logp-logp[:,:,baseline].mean(2,keepdims=True))[:,keepf][:,:,keept])
        power.append(logp[:,keepf][:,:,keept])
        band_db=np.stack([10*np.log10(np.maximum(p[:,(freq>=lo)&(freq<hi)].mean(1),1e-12)) for lo,hi in BANDS],axis=1)
        pre.append(band_db[:,:,baseline].mean(2).ravel());post.append(band_db[:,:,active].mean(2).ravel())
    return dict(relative=np.array(maps),absolute=np.array(power),pre=np.array(pre),post=np.array(post),
                freq=freq[keepf],time=t[keept],labels=y,blocks=blocks,trial_ids=ids,channels=raw['channel_ids'],fs=fs,
                alignment=np.array(alignment),pre_centers=t[baseline],post_centers=t[active],window_s=256/fs)

def blocked_direction_test(post,delta,y,blocks,channels):
    # Compare block-fixed-effects null against block + categorical direction.
    Y=np.c_[post,delta];unique=np.unique(blocks)
    B=(blocks[:,None]==unique).astype(float);B/=np.sqrt((B**2).sum(0))
    Y0=Y-B@(B.T@Y);sse0=(Y0**2).sum(0);df1=7;df2=len(y)-len(unique)-df1
    def statistic(labels):
        D=(labels[:,None]==np.arange(1,8)).astype(float);D-=B@(B.T@D)
        assert np.linalg.matrix_rank(D)==7
        q,_=np.linalg.qr(D,mode='reduced');explained=np.sum((q.T@Y0)**2,axis=0)
        residual=np.maximum(sse0-explained,1e-12)
        return (explained/df1)/(residual/df2),explained/(explained+residual)
    observed,eta=statistic(y)
    # Independent least-squares reconstruction of observed F statistics.
    design=np.c_[B,(y[:,None]==np.arange(1,8)).astype(float)]
    residual=Y-design@np.linalg.lstsq(design,Y,rcond=None)[0]
    reference=((sse0-np.sum(residual**2,axis=0))/df1)/(np.sum(residual**2,axis=0)/df2)
    np.testing.assert_allclose(observed,reference,rtol=1e-8,atol=1e-10)
    rng=np.random.default_rng(11132026);null=np.empty((N_PERM,Y.shape[1]))
    for k in range(N_PERM):
        perm=y.copy()
        for b in unique:
            idx=np.flatnonzero(blocks==b);perm[idx]=rng.permutation(perm[idx])
        null[k]=statistic(perm)[0]
    p=(1+np.sum(null>=observed,axis=0))/(N_PERM+1)
    adjusted=(1+np.sum(null.max(1)[:,None]>=observed,axis=0))/(N_PERM+1)
    q=bh_fdr(p)
    rows=[]
    for i in range(Y.shape[1]):
        c,b=divmod(i%36,6)
        rows.append(dict(response='post_power' if i<36 else 'post_minus_pre',channel_id=int(channels[c]),
            band_hz=f'{BANDS[b][0]}-{BANDS[b][1]}',F=float(observed[i]),df_direction=df1,df_residual=df2,
            partial_eta_squared=float(eta[i]),permutation_p=float(p[i]),BH_q_72=float(q[i]),
            maxF_FWER_p_72=float(adjusted[i]),significant_FWER=bool(adjusted[i]<.05)))
    return rows,null

def task_change_test(delta,y,blocks,channels):
    unique=np.unique(blocks);cluster=np.full((len(unique),8,36),np.nan)
    for i,b in enumerate(unique):
        for d in range(8):
            m=(blocks==b)&(y==d)
            if m.any():cluster[i,d]=delta[m].mean(0)
    count=np.isfinite(cluster).sum(0);values=np.nan_to_num(cluster);ss=np.sum(values**2,axis=0)
    def tstat(sums):
        variance=np.maximum((ss-sums**2/count)/(count-1),1e-12)
        return (sums/count)/np.sqrt(variance/count)
    observed=tstat(values.sum(0))
    rng=np.random.default_rng(13112026);signs=rng.choice([-1,1],size=(N_PERM,len(unique)))
    # One sign per whole block, shared across direction/channel/band cells.
    perm=tstat(np.einsum('rb,bdf->rdf',signs,values))
    p=(1+np.sum(np.abs(perm)>=np.abs(observed),axis=0))/(N_PERM+1)
    adjusted=(1+np.sum(np.max(np.abs(perm),axis=(1,2))[:,None,None]>=np.abs(observed)[None,:,:],axis=0))/(N_PERM+1)
    q=bh_fdr(p.ravel()).reshape(8,36);effect=np.nanmean(cluster,axis=0);rows=[]
    for d in range(8):
        for j in range(36):
            c,b=divmod(j,6)
            rows.append(dict(direction_deg=d*45,channel_id=int(channels[c]),band_hz=f'{BANDS[b][0]}-{BANDS[b][1]}',
                blocks=int(count[d,j]),mean_change_db=float(effect[d,j]),t=float(observed[d,j]),
                paired_block_signflip_p=float(p[d,j]),BH_q_288=float(q[d,j]),maxT_FWER_p_288=float(adjusted[d,j]),
                significant_FWER=bool(adjusted[d,j]<.05)))
    return rows,effect.reshape(8,6,6),adjusted.reshape(8,6,6)

def finish(fig,name,pdf=True):
    fig.canvas.draw()
    fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15)
    if pdf:fig.savefig(OUT/(name+'.pdf'),facecolor='white',bbox_inches='tight',pad_inches=.15)

def draw_maps(data):
    labels=data['labels'];channels=data['channels'];frequency=data['freq'];times=data['time']
    relative=np.array([data['relative'][labels==d].mean(0) for d in range(8)])
    absolute=np.array([data['absolute'][labels==d].mean(0) for d in range(8)])
    np.savez_compressed(OUT/'direction_mean_spectrograms.npz',relative_db=relative,absolute_db=absolute,
                        frequencies_hz=frequency,time_s=times,channels=channels,counts=np.bincount(labels,minlength=8))
    overview=relative.mean(1);limit=max(1,np.ceil(np.quantile(np.abs(overview),.99)*2)/2)
    for kind,values,lo,hi,cmap,units in [
        ('relative',overview,-limit,limit,'RdBu_r','Power change (dB vs pre-event reference)'),
        ('absolute',absolute.mean(1),float(np.quantile(absolute.mean(1),.01)),float(np.quantile(absolute.mean(1),.99)),'viridis','Mean log PSD (dB; raw ADC²/Hz)')]:
        fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained',sharex=True,sharey=True)
        for d,ax in enumerate(axs.flat):
            im=ax.pcolormesh(times,frequency,values[d],shading='auto',vmin=lo,vmax=hi,cmap=cmap,rasterized=True)
            ax.axvline(0,color='k',ls='--',lw=.8)
            ax.set(title=f'{d*45}° | n = {sum(labels==d)}',xlim=(times.min(),times.max()),ylim=(4,200),xticks=[-.5,0,.5],yticks=[10,50,100,150,200])
            if d%4==0:ax.set_ylabel('Frequency (Hz)')
            if d//4==1:ax.set_xlabel('Time from target onset (s)')
        fig.colorbar(im,ax=axs,label=units,fraction=.022,pad=.02)
        title='Event-related power change' if kind=='relative' else 'Direction-aligned power spectrogram'
        fig.suptitle(f'1111 CO8 run-009 | {title}\nEqual-weight mean across 6 channels; dashed line = target onset',fontsize=17)
        finish(fig,'LFP_Eight_Direction_Spectrogram_'+kind.title());plt.close(fig)
    channel_limit=max(1,np.ceil(np.quantile(np.abs(relative),.99)*2)/2)
    with PdfPages(OUT/'LFP_Six_Channels_Direction_Spectrograms.pdf') as pdf:
        for c,ch in enumerate(channels):
            fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained',sharex=True,sharey=True)
            for d,ax in enumerate(axs.flat):
                im=ax.pcolormesh(times,frequency,relative[d,c],shading='auto',cmap='RdBu_r',vmin=-channel_limit,vmax=channel_limit,rasterized=True)
                ax.axvline(0,color='k',ls='--',lw=.8);ax.set(title=f'{d*45}° | n = {sum(labels==d)}',ylim=(4,200),xticks=[-.5,0,.5],yticks=[10,50,100,150,200])
                if d%4==0:ax.set_ylabel('Frequency (Hz)')
                if d//4==1:ax.set_xlabel('Time from target onset (s)')
            fig.colorbar(im,ax=axs,label='Power change (dB vs pre-event reference)',fraction=.022,pad=.02)
            fig.suptitle(f'Wire channel {ch} (0-based) | Eight target directions\nPre-event reference contains ongoing task activity, not rest',fontsize=17)
            finish(fig,f'LFP_Channel_{ch}_Direction_Spectrogram',pdf=False);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    # Compact atlas shows all 48 channel/direction combinations with one common color scale.
    fig,axs=plt.subplots(6,8,figsize=(24,15),layout='constrained',sharex=True,sharey=True)
    for c,ch in enumerate(channels):
        for d in range(8):
            ax=axs[c,d];im=ax.pcolormesh(times,frequency,relative[d,c],shading='auto',vmin=-channel_limit,vmax=channel_limit,cmap='RdBu_r',rasterized=True)
            ax.axvline(0,color='k',lw=.6,ls='--');ax.tick_params(labelsize=8);ax.set(xticks=[-.5,0,.5],yticks=[10,100,200])
            if c==0:ax.set_title(f'{d*45}° (n={sum(labels==d)})',fontsize=11)
            if d==0:ax.set_ylabel(f'Ch {ch}\nFrequency (Hz)',fontsize=10)
            if c==5:ax.set_xlabel('Time (s)',fontsize=10)
    fig.colorbar(im,ax=axs,label='Power change (dB vs pre-event reference)',fraction=.012,pad=.012)
    fig.suptitle('1111 CO8 run-009 | Six channels × eight directions | Target onset = 0 s',fontsize=20)
    finish(fig,'LFP_Direction_Channel_Spectrogram_Atlas');plt.close(fig)
    return dict(overview_clim_db=[-limit,limit],channel_clim_db=[-channel_limit,channel_limit])

def draw_statistics(data,direction_rows,response_rows,effect,adjusted):
    channels=data['channels']
    fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    limit=max(.5,np.ceil(np.max(np.abs(effect))*2)/2)
    for b,ax in enumerate(axs.flat):
        a=effect[:,:,b].T;im=ax.imshow(a,aspect='auto',vmin=-limit,vmax=limit,cmap='RdBu_r')
        for c in range(6):
            for d in range(8):
                marker='*' if adjusted[d,c,b]<.05 else ''
                ax.text(d,c,f'{a[c,d]:.1f}{marker}',ha='center',va='center',fontsize=9,color='white' if abs(a[c,d])>.65*limit else '#222')
        ax.set(title=f'{BANDS[b][0]}–{BANDS[b][1]} Hz',xticks=range(8),xticklabels=np.arange(8)*45,yticks=range(6),yticklabels=channels,xlabel='Target direction (deg)',ylabel='Wire channel (0-based)')
    fig.colorbar(im,ax=axs,label='Mean post-minus-pre log power (dB)',fraction=.022,pad=.02)
    fig.suptitle('Event-related band-power changes by direction\n* Block sign-flip max-T corrected p < 0.05 across 288 tests; pre-event reference is not rest',fontsize=15)
    finish(fig,'LFP_Direction_Band_Power_Response');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for response,ax,title in zip(['post_power','post_minus_pre'],axs,['Post-event power','Post-minus-pre power change']):
        selected=[r for r in direction_rows if r['response']==response]
        p=np.array([r['maxF_FWER_p_72'] for r in selected]).reshape(6,6)
        maximum=max(2,float(-np.log10(np.min([r['maxF_FWER_p_72'] for r in direction_rows]))))
        im=ax.imshow(-np.log10(p),aspect='auto',cmap='viridis',vmin=0,vmax=maximum)
        for c in range(6):
            for b in range(6):
                label=f'p={p[c,b]:.4f}*' if p[c,b]<.05 else 'n.s.'
                ax.text(b,c,label,ha='center',va='center',fontsize=8 if p[c,b]<.05 else 10,
                        color='#222' if -np.log10(p[c,b])>.65*maximum else 'white')
        ax.set(title=title,xticks=range(6),xticklabels=[f'{lo}–{hi}' for lo,hi in BANDS],yticks=range(6),yticklabels=channels,xlabel='Frequency band (Hz)',ylabel='Wire channel (0-based)')
    fig.colorbar(im,ax=axs,label='−log10(corrected permutation p)',fraction=.025,pad=.02)
    fig.suptitle('Direction selectivity | block-adjusted categorical direction effect\nWithin-block permutations; max-F correction across all 72 tests',fontsize=15)
    finish(fig,'LFP_Direction_Selectivity_Statistics');plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    degrees=np.arange(8)*45
    for c,ax in enumerate(axs.flat):
        values=data['post'][:,c*6+5]
        means=np.array([values[data['labels']==d].mean() for d in range(8)])
        sem=np.array([values[data['labels']==d].std(ddof=1)/np.sqrt(sum(data['labels']==d)) for d in range(8)])
        item=next(r for r in direction_rows if r['response']=='post_power' and r['channel_id']==data['channels'][c] and r['band_hz']=='105-180')
        color='#c44e22' if item['significant_FWER'] else '#2176ae'
        ax.errorbar(degrees,means,yerr=sem,fmt='-o',color=color,ms=4,capsize=3)
        ax.set(title=f'Ch {data["channels"][c]} | corrected p = {item["maxF_FWER_p_72"]:.4f}',
               xticks=degrees,xlabel='Target direction (deg)',ylabel='Post-event log band PSD (dB)')
        ax.grid(alpha=.15)
    fig.suptitle('105–180 Hz direction tuning | mean ± SEM across trials\nRaw ADC²/Hz in dB; separate y scales per channel; p corrected across all 72 tests',fontsize=15)
    finish(fig,'LFP_High_Gamma_Direction_Tuning');plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[BASE/'unpacked_lfp.npz',BASE/'decoding_features.npz',BASE/'reconstructed_trials.csv']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    data=tf_data();post=data['post'];delta=post-data['pre']
    print('Trials',len(post),'direction counts',np.bincount(data['labels'],minlength=8),'TF shape',data['relative'].shape,flush=True)
    with threadpool_limits(limits=1):
        drows,null=blocked_direction_test(post,delta,data['labels'],data['blocks'],data['channels'])
        rrows,effect,adj=task_change_test(delta,data['labels'],data['blocks'],data['channels'])
    table('direction_selectivity_tests.csv',drows);table('event_related_power_tests.csv',rrows)
    trial_features=[]
    for i,tid in enumerate(data['trial_ids']):
        for j in range(36):
            c,b=divmod(j,6);trial_features.append(dict(trial=int(tid),block=int(data['blocks'][i]),direction_deg=int(data['labels'][i]*45),
                channel_id=int(data['channels'][c]),band_hz=f'{BANDS[b][0]}-{BANDS[b][1]}',pre_db=float(data['pre'][i,j]),post_db=float(post[i,j]),change_db=float(delta[i,j])))
    table('trial_band_power.csv',trial_features)
    np.savez_compressed(OUT/'trial_spectrograms.npz',relative_db=data['relative'].astype(np.float32),absolute_db=data['absolute'].astype(np.float32),
        post_db=post,pre_db=data['pre'],trial_ids=data['trial_ids'],directions=data['labels'],blocks=data['blocks'],time_s=data['time'],frequencies_hz=data['freq'],channel_ids=data['channels'])
    np.savez_compressed(OUT/'direction_permutation_null.npz',F=null)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':['Arial','DejaVu Sans'],'pdf.fonttype':42})
    color=draw_maps(data);draw_statistics(data,drows,rrows,effect,adj)
    significant_direction=[r for r in drows if r['significant_FWER']]
    significant_response=[r for r in rrows if r['significant_FWER']]
    summary=dict(trials=len(post),directions_degrees=list(range(0,360,45)),direction_counts=np.bincount(data['labels'],minlength=8),channels=data['channels'],
        sample_rate_hz_estimated=data['fs'],fft_points=256,frequency_resolution_hz=data['fs']/256,window_s=data['window_s'],hop_s=20/data['fs'],
        reference_STFT_centers_s=data['pre_centers'],post_STFT_centers_s=data['post_centers'],max_sample_alignment_error_s=float(np.max(np.abs(data['alignment']))),
        display=data['time'][[0,-1]],color_limits=color,
        significant_direction=significant_direction,significant_task_related_changes=significant_response,
        direction_method='Trial-level mean log band PSD; fixed effects for block plus 7 categorical direction contrasts. 1999 direction-label permutations restricted within block. Joint max-F correction across 6 channels x 6 bands x 2 response definitions=72 tests. F checked independently against full-design least squares.',
        response_method='Trial post-minus-pre mean log band PSD; one observation per direction per block. 1999 paired sign flips with shared sign per whole block; two-sided max-|t| correction across 8 directions x 6 channels x 6 bands=288 tests. Sign-flip interpretation assumes symmetric block-level differences under the null.',
        limitations=['Pre-event reference contains previous-trial ongoing behavior because ITI=0; not REST and not a causal movement-onset contrast.',
            'Target-locked spectra include sensory, preparation and movement contributions; they cannot establish motor intention or exclude EMG/motion artifact.',
            'STFT uses about 246 ms windows (~4.07 Hz resolution), so <4 Hz dynamics are not resolved; event-time bins are not independent statistical observations.',
            'All time-frequency pixels are descriptive: stars apply only to predeclared band/time aggregate tests, not pixelwise significance.',
            'Frequency values and target alignment inherit estimated acquisition clock and uncertain absolute latency; ADC-to-microvolt scale unknown.',
            'Same exploratory recording as earlier decoding; six-channel averaging can mask opposite or channel-specific effects. Separate channel maps are supplied.'],
        documentation=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html'],source_sha256=before)
    dump('spectrogram_results.json',summary)
    DELIVERY.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ('.png','.pdf','.csv','.json'):shutil.copy2(p,DELIVERY/p.name)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    print('Significant directional effects:',significant_direction,flush=True)
    print('Significant phase changes:',len(significant_response),'of',len(rrows),flush=True)
    print('DELIVERY',DELIVERY,flush=True)

if __name__=='__main__':main()
