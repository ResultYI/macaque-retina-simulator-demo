"""Plot saved group states only; no simulator or observations are loaded."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(r'D:\macaque retinal simulator/reports/GIN_69_6_HC2_BC1_SUPPORT_V2')
z=np.load(r/'executions/joint_short_01/forward/standard/RESPONSE.npz',allow_pickle=False)
t=z['time_seconds']*1000
fig,axes=plt.subplots(2,3,figsize=(13,7));a=axes.ravel()
a[0].step([0,13.333333,13.333333,26.666667,26.666667,50],[1,1,1.2,1.2,.8,.8],where='post')
a[0].set_ylabel('Native input / local gray');a[0].set_title('Controlled engineering input')
for pop in ['midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC']:
    v=z['BC__'+pop+'__voltage'].mean(0);a[1].plot(t,(v-v[0])*1e6,label=pop.replace('_BC',''))
a[1].set_ylabel('Delta V (uV), group mean');a[1].set_title('Five BC populations');a[1].legend(fontsize=6)
for pop in ['same_ON_midget_AC_v1','same_OFF_midget_AC_v1','cross_OFF_ON_parasol_AC_v1','cross_ON_OFF_parasol_AC_v1']:
    v=z['AC__'+pop+'__voltage'].mean(0);a[2].plot(t,(v-v[0])*1e6,label=pop.replace('_v1',''))
a[2].set_ylabel('Delta V (uV), group mean');a[2].set_title('Four effective AC pools');a[2].legend(fontsize=5)
v=z['RGC__parasol_RGC_OFF__voltage'][3];a[3].plot(t,(v-v[0])*1e6)
a[3].set_ylabel('Delta V (uV)');a[3].set_title('Target parent row7730 membrane')
for port in ['excitatory_conductance','inhibitory_conductance']:
    a[4].plot(t,z['RGC__parasol_RGC_OFF__'+port][3]*1e9,label=port.split('_')[0])
a[4].set_ylabel('Conductance (nS)');a[4].set_title('Actual target E/I');a[4].legend(fontsize=7)
a[5].plot(t,z['RGC__parasol_RGC_OFF__expected_binary_count_rate'][3])
a[5].set_ylabel('Expected group event rate (Hz)');a[5].set_title('Model event rate; no fitted readout')
for ax in a:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2)
fig.suptitle('69#6 joint H1/BC engineering smoke test (50ms)\nSaved states; representative groups; fixed exterior gray; not physiological RF validation',fontsize=11)
fig.tight_layout()
for suffix in ['png','svg']:fig.savefig(r/('JOINT_SHORT_STATES.'+suffix),dpi=170)
plt.close(fig)
print('Saved actual-state plots; no model run')
