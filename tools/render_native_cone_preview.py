"""Plot saved native aperture geometry and actual saved optical captures."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


def discs(ax,xy,diameters,colors):
    collection=EllipseCollection(diameters,diameters,np.zeros(len(xy)),units='xy',
        offsets=xy,transOffset=ax.transData,facecolors=colors,edgecolors='none')
    ax.add_collection(collection);ax.set_aspect('equal')


def main(folder):
    with np.load(folder/'preview_data.npz',allow_pickle=False) as z:
        xy=z['xy_mm'];diameters=z['apertures_um'];types=z['types'];rates=z['rates_Rstar_s'];rgb=z['rgb']
    palette=np.array([[.84,.17,.23],[.08,.57,.34],[.18,.38,.81]])
    kinds=np.array([{'L':0,'M':1,'S':2}[k] for k in types]);colors=palette[kinds]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    axes[0].imshow(rgb,extent=(-6,6,-6,6))
    discs(axes[1],xy/.223,diameters/223,colors)
    norm=Normalize(0,float(rates.max()));cmap=plt.get_cmap('viridis')
    discs(axes[2],xy/.223,diameters/223,cmap(norm(rates)))
    for ax,title in zip(axes,['A  Input frame','B  L/M/S Cone mosaic','C  Cone photon catches']):
        ax.set(xlim=(-6,6),ylim=(-6,6),xlabel='x (deg)',ylabel='y (deg)',title=title)
        ax.set_xticks([-6,-3,0,3,6]);ax.set_yticks([-6,-3,0,3,6])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[2],label='R*/Cone/s',shrink=.8)
    fig.savefig(folder/'native_cone_preview.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,ecc in zip(axes,[0.,3.,5.8]):
        local=(xy-np.array([ecc*.223,0]))*1000
        keep=np.all(np.abs(local)<30,axis=1)
        discs(ax,local[keep],diameters[keep],colors[keep])
        ax.set(xlim=(-20,20),ylim=(-20,20),title=f'{ecc:g}°',xlabel='Local x (µm)',ylabel='Local y (µm)')
    fig.savefig(folder/'native_diameter_same_scale.png',dpi=300);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder.resolve())
