"""Plot only already-saved69#6 data with the existing base plotting environment."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.io import loadmat
ROOT=Path(r'D:\macaque retinal simulator')
BASE=ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf8')
def main():
    counts=read(BASE/'prepared/STATIC_COUNTS.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=np.load(BASE/'forward/standard/RESPONSE.npz',allow_pickle=False);tm=data['time_seconds'];row=counts['target_local_row'];prefix='RGC__parasol_RGC_OFF__'
    traces={name:data[prefix+key][row] for name,key in [('V_V','voltage'),('gE_S','excitatory_conductance'),('gI_S','inhibitory_conductance'),('expected_hazard_Hz','expected_hazard'),('expected_binary_count_rate_Hz','expected_binary_count_rate')]}
    with (BASE/'TARGET_RESPONSE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','source_frame_zero_based',*traces])
        for i,time_s in enumerate(tm):w.writerow([time_s,751+int(np.floor(time_s*150)),*(a[i] for a in traces.values())])
    response_summary={key:dict(initial=float(v[0]),minimum=float(v.min()),maximum=float(v.max()),final=float(v[-1])) for key,v in traces.items()}
    write(BASE/'TARGET_RESPONSE_SUMMARY.json',dict(target=counts['target_group_id'],members=counts['target_members'],fields=response_summary,simulation_s=.05,spikes_are_group_objects=True))
    fig,axes=plt.subplots(6,1,figsize=(10,12),sharex=True,constrained_layout=True)
    with np.load(BASE/'background/MOVIE_NATIVE_INPUTS.npz',allow_pickle=False) as z:drive=z['live_native_drive']
    axes[0].step(np.r_[np.arange(8)*1000/150,50],np.r_[drive.mean(1),drive.mean(1)[-1]],where='post');axes[0].set_ylabel('Cone input\nnative equivalent')
    for key,label in [('STATE__HC__h1_h2_effective_v1__h1','H1'),('STATE__HC__h1_h2_effective_v1__h2','H2')]:axes[1].plot(tm*1000,data[key].mean(0),label=label)
    axes[1].set_ylabel('Local HC mean\nnative state');axes[1].legend()
    for pop in ('diffuse_BC_ON','diffuse_BC_OFF'):axes[2].plot(tm*1000,data['BC__'+pop+'__voltage'].mean(0)*1000,label=pop)
    axes[2].set_ylabel('Local BC mean\nmV');axes[2].legend(fontsize=8)
    for name in data.files:
        if name.startswith('STATE__') and 'FunctionalAmacrineVoltageGate_v1__V' in name:axes[3].plot(tm*1000,data[name].mean(0)*1000,label=name.split('__')[1])
    axes[3].set_ylabel('Local AC mean\nmV');axes[3].legend(fontsize=7,ncol=2)
    axes[4].plot(tm*1000,traces['V_V']*1000,label='OFF-parasol V');axes[4].set_ylabel('Target V\nmV')
    axes[5].plot(tm*1000,traces['expected_hazard_Hz'],label='Expected hazard');axes[5].plot(tm*1000,traces['expected_binary_count_rate_Hz'],label='Expected binary count rate');axes[5].set_ylabel('Target rate\nHz');axes[5].legend();axes[5].set_xlabel('Time from declared source live start (ms)')
    for ax in axes:ax.set_xlim(0,50);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].set_title('69#6 radial conditional candidate: actual source movie, 4 mm pupil\nLocal pool means are display summaries, not identified inputs to this RGC')
    fig.savefig(BASE/'SHORT_RESPONSE.png',dpi=180);fig.savefig(BASE/'SHORT_RESPONSE.svg');plt.close(fig)
    frame=loadmat(BASE/'SOURCE_BANK.mat')['source_rgb'][:,:,:,1];fig,ax=plt.subplots(figsize=(5,5),constrained_layout=True);ax.imshow(frame)
    binding=read(BASE/'INPUT_APPROXIMATION_MANIFEST.json');ax.plot(binding['source_RF_fractional_column'],binding['source_RF_fractional_row'],'r+',ms=14);ax.set_title('Source frame751; author fitted RF center');fig.savefig(BASE/'SOURCE_FRAME751.png',dpi=150);plt.close(fig)
    data.close()

if __name__=='__main__':main()
