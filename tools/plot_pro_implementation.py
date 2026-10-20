"""One final scientific figure from saved Pro implementation results only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True);a=p.parse_args();root=a.reports.resolve()
    names={'midget_RGC_ON':'ON midget','midget_RGC_OFF':'OFF midget','parasol_RGC_OFF':'OFF parasol','small_bistratified_S_ON':'Small bistratified'}
    fig,axes=plt.subplots(3,2,figsize=(13,12));colors={'center05':'#6877a5','center20':'#172c50','ring20':'#c16f24','LM20':'#41835d'}
    for ax,(cid,label) in zip(axes.flat,names.items()):
        d=root/cid;r=json.loads((d/'result.json').read_bytes())
        if r['status']!='COMPLETE_OWN_TYPE_MAIN_ROUTE_ACCEPTANCE':raise ValueError('Incomplete own-type result: '+cid)
        with np.load(d/'response_arrays.npz',allow_pickle=False) as arrays:
            for name in ('center05','center20','ring20','LM20'):
                key=name+'.expected_delta_Hz'
                if key not in arrays:continue
                focus=r['responses'][name]['focus_cell_index'];v=arrays[key][focus]
                text={'center05':'Center 5%','center20':'Center 20%','ring20':'Ring 20%','LM20':'L/M center 20%'}[name]
                if cid=='small_bistratified_S_ON' and name!='LM20':text='S '+text.lower()
                ax.plot(np.arange(len(v)),v,color=colors[name],lw=1.5,label=text)
        ax.axvspan(50,100,color='#f3dc93',alpha=.35);ax.axvspan(150,200,color='#b3c7e3',alpha=.3)
        ax.axhline(0,color='#888888',lw=.6);ax.set_xlabel('Time after own checkpoint (ms)');ax.set_ylabel('Expected binary count rate difference (Hz)')
        decision='Passes relative checks' if r['all_checks_passed'] else ('Candidate retained: dark and spatial/spectral criteria unmet' if not r['passed_checks']['bright_dark'] else 'Candidate retained: spatial criterion unmet')
        ax.set_title(label+'\n'+decision,fontsize=10);ax.legend(fontsize=8,frameon=False)
    r=json.loads((root/'rod_fixed_dark_regression/result.json').read_bytes());ax=axes[2,0]
    x=np.array([1.,5.,20.]);v=np.array([r['backgrounds'][str(t)]['initial_voltage_mV']['mean'] for t in x])
    ax.plot(x,v,'o-',color='#172c50');ax.set_xscale('log');ax.set_xticks(x,labels=['1','5','20']);ax.set_xlabel('Actual rod background (R*/s)');ax.set_ylabel('Derived initial rod voltage (mV)')
    for t,y in zip(x,v):ax.annotate('q = %.5f'%r['backgrounds'][str(t)]['initial_release']['mean'],(t,y),xytext=(4,8),textcoords='offset points',fontsize=9)
    ax.set_title('Fixed dark current reference = 0 A',fontsize=11);ax.margins(x=.15,y=.25)
    ax=axes[2,1]
    with np.load(root/'rod_fixed_dark_regression/flash_gray_deltas.npz',allow_pickle=False) as data:
        t=(data['time']-data['time'][0])*1000
        ax.plot(t,data['rod_membrane.voltage.delta'].mean(axis=0)*1000,color='#172c50',label='Rod membrane')
        other=ax.twinx();other.plot(t,data['rod_BC.voltage.delta'].mean(axis=0)*1e6,color='#c16f24',label='RBC')
    ax.axvspan(50,150,color='#f3dc93',alpha=.35);ax.set_xlabel('Time after own checkpoint (ms)');ax.set_ylabel('Rod voltage difference (mV)',color='#172c50');other.set_ylabel('RBC voltage difference (µV)',color='#c16f24')
    ax.set_title('5→6 R*/s rod-only flash minus own gray',fontsize=11)
    for ax in axes.flat:ax.spines[['top']].set_visible(False);ax.grid(alpha=.16)
    fig.suptitle('Pro parameter implementation — actual 34-component shared main',fontsize=15)
    fig.text(.5,.025,'Each type uses its own candidate and matched gray. 100 ms wash is not full adaptation.\nEngineering relative-function checks; no animal amplitude fit or physiological calibration.',ha='center',fontsize=10)
    fig.tight_layout(rect=(.015,.065,.985,.965));fig.savefig(root/'pro_parameter_implementation.png',dpi=170);plt.close(fig)
    print(root/'pro_parameter_implementation.png')


if __name__=='__main__':main()
