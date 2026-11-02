"""Paper-family handcrafted features; extraction details are explicit adaptations.

256-bin HSV histogram + 24 Haar statistics + 5 edge-filter counts. Each block
is L2 normalized separately, then concatenated; no labels or deep models.
"""
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv


def descriptor(path):
    rgb=np.asarray(Image.open(path).convert('RGB'),dtype=float)/255.
    hsv=rgb_to_hsv(rgb)
    color=np.histogramdd(hsv.reshape(-1,3),bins=(8,8,4),range=((0,1),(0,1),(0,1)))[0].ravel()
    color/=color.sum()
    y=.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]
    # Three 2-D orthonormal Haar levels, all four subbands at each level.
    approx=y;texture=[]
    for _ in range(3):
        h,w=approx.shape;approx=approx[:h-h%2,:w-w%2]
        a,b,c,d=approx[::2,::2],approx[::2,1::2],approx[1::2,::2],approx[1::2,1::2]
        bands=[(a+b+c+d)/2,(a+b-c-d)/2,(a-b+c-d)/2,(a-b-c+d)/2]
        for band in bands:texture.extend([float(band.mean()),float(band.std())])
        approx=bands[0]
    h,w=y.shape;y=y[:h-h%2,:w-w%2]
    blocks=np.stack([y[::2,::2],y[::2,1::2],y[1::2,::2],y[1::2,1::2]],axis=-1)
    q=np.sqrt(2.)
    filters=np.array([[1,1,-1,-1],[q,0,0,-q],[1,-1,1,-1],[0,q,-q,0],[2,-2,-2,2]])
    response=np.abs(blocks@filters.T)/4
    strongest=response.argmax(axis=-1);active=response.max(axis=-1)>.1
    edge=np.bincount(strongest[active],minlength=5)/strongest.size
    blocks=[color,np.asarray(texture),edge]
    result=np.concatenate([b/max(np.linalg.norm(b),1e-15) for b in blocks])
    assert result.shape==(285,) and np.isfinite(result).all()
    return result
