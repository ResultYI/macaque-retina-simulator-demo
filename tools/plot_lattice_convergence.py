"""Plot frozen geometry and complete convergence history, without advancing it."""
from pathlib import Path
import argparse,json,os
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/lattice_coverage_repair_20261015/continuation'
os.environ.setdefault('MPLCONFIGDIR',str(REPORT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data_directory');args=p.parse_args()
    data=(ROOT/args.data_directory).resolve();assert data.is_relative_to(ROOT)
    g=json.loads((data/'generation.json').read_bytes());manifest=json.loads((data/'manifest.json').read_bytes())
    with np.load(data/'coordinates.npz',allow_pickle=False) as z:xy=-z['selected_raw_um']/1000
    nlegacy=manifest['exact_initialization']['coverage_metadata']['legacy_cells_preserved']
    radius=manifest['radius_um']/1000;t=g['trace'];iteration=np.array([x['iteration'] for x in t])
    fig=plt.figure(figsize=(12,7),constrained_layout=True);gs=fig.add_gridspec(2,2,width_ratios=[1,1.3])
    ax=fig.add_subplot(gs[:,0]);qax=fig.add_subplot(gs[0,1]);maxis=fig.add_subplot(gs[1,1])
    ax.scatter(xy[:nlegacy,0],xy[:nlegacy,1],s=.12,c='#626d78',linewidths=0,rasterized=True)
    ax.scatter(xy[nlegacy:,0],xy[nlegacy:,1],s=.3,c='#168bd2',linewidths=0,rasterized=True)
    ax.add_patch(Ellipse((0,0),2*radius,2*1.2247*radius,fill=False,color='#101820',lw=.8))
    ax.add_patch(Rectangle((-.512,-.512),1.024,1.024,fill=False,color='#bc5c00',lw=.8,ls='--'))
    ax.set(aspect='equal',xlabel='RF/input x (mm)',ylabel='RF/input y (mm)',title=f'Saved snapshot: step {g["selected_iteration"]:,}\nAll {len(xy):,} points; unchanged ellipse')
    qax.plot(iteration,[x['q_0_8'] for x in t],color='#176c9a',lw=1,label='q: 0.8th percentile (Hazen)')
    qax.axhline(.84,color='#9b253f',lw=1,ls='--',label='Original q threshold: 0.84')
    qax.axvline(g['selected_iteration'],color='#64727a',lw=.8,ls=':')
    qax.set(ylabel='Triangle quality q = 2r/R',title='Quality criterion also requires source width history')
    qax.legend(fontsize=8,loc='lower right')
    maxis.semilogy(iteration,[x.get('source_interior_p30_motion_um',np.nan) for x in t],color='#176c9a',lw=1,label='Source internal p30: sqrt(0.2)*|F|')
    maxis.semilogy(iteration,[x['actual_max_movement_um'] for x in t],color='#8f989f',lw=.7,label='True maximum displacement (diagnostic)')
    maxis.semilogy(iteration,[x['actual_p30_movement_um'] for x in t],color='#4e9e8a',lw=.7,label='True p30 displacement (diagnostic)')
    maxis.axhline(.000215,color='#9b253f',ls='--',lw=1,label='Source p30 tolerance: 0.000215 um')
    maxis.set(xlabel='Cumulative accepted step',ylabel='Source statistic / true displacement (um)',title='OR branch: source internal p30; true maximum is diagnostic')
    maxis.legend(fontsize=8,loc='upper right')
    for a in (ax,qax,maxis):a.grid(alpha=.16)
    fig.suptitle(f'Complete coverage | Stop: {g["stop_reason"]} | No physiological validation',fontsize=11)
    fig.savefig(REPORT/'final_geometry_and_trends.png',dpi=180);plt.close(fig)
    print(REPORT/'final_geometry_and_trends.png')
