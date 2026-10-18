"""Render frozen coordinates only; no generation, physics or neural work."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/lattice_coverage_repair_20261015'
os.environ.setdefault('MPLCONFIGDIR',str(REPORT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle
import numpy as np

def main():
    data=ROOT/'data/atlas/isetbio_macaque_cones_coveragefix_v1_20261015'
    with np.load(data/'coordinates.npz',allow_pickle=False) as a:initial=-a['initial_raw_um']/1000;selected=-a['selected_raw_um']/1000
    v=json.loads((REPORT/'verification.json').read_bytes());meta=v['approved_initialization_coverage']['metadata']
    legacy=meta['legacy_cells_preserved'];R=meta['radius_um_unchanged']/1000
    fig,axes=plt.subplots(1,2,figsize=(10,6),constrained_layout=True)
    for ax,xy,title in zip(axes,[initial,selected],['Complete initialization','Saved relaxation snapshot']):
        ax.scatter(xy[:legacy,0],xy[:legacy,1],s=.12,c='#626d78',linewidths=0,rasterized=True,label='Legacy source samples')
        ax.scatter(xy[legacy:,0],xy[legacy:,1],s=.3,c='#168bd2',linewidths=0,rasterized=True,label='Samples from omitted candidates')
        ax.add_patch(Ellipse((0,0),2*R,2*1.2247*R,fill=False,color='#101820',lw=.8))
        ax.add_patch(Rectangle((-.512,-.512),1.024,1.024,fill=False,color='#bc5c00',lw=.8,ls='--'))
        ax.set(xlim=(-.84,.84),ylim=(-1.02,1.02),aspect='equal',xlabel='RF/input x (mm)',ylabel='RF/input y (mm)',title=title)
        ax.grid(alpha=.12)
    axes[0].axhline(-.7993700944595143,color='#ba3344',lw=.7,ls=':')
    axes[0].axhline(.7993700944595143,color='#ba3344',lw=.7,ls=':')
    axes[0].legend(loc='upper right',fontsize=7,markerscale=8,framealpha=.9)
    fig.suptitle(f'Same ellipse, density, hex phase and seed | {legacy:,} retained + {len(initial)-legacy:,} sampled = {len(initial):,}',fontsize=10)
    fig.supxlabel('Dashed square: unchanged declared query support.  Geometry only; no default adoption.\n300-step budget stop; original quality/movement conditions not reached.',fontsize=8)
    fig.savefig(REPORT/'coverage_and_saved_geometry.png',dpi=180)
    plt.close(fig)
    print(REPORT/'coverage_and_saved_geometry.png')
if __name__=='__main__':main()
