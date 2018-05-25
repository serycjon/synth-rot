from __future__ import print_function

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize
import scipy.optimize
import matplotlib.pyplot as plt
import cv2

# https://tex.stackexchange.com/a/144691
### Gaussian smoother from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/ for getting a nice polygon

def smoothListGaussian(list,degree=5):
    window=degree*2-1
    weight=np.array([1.0]*window)
    weightGauss=[]
    for i in range(window):
        i=i-degree+1
        frac=i/float(window)
        gauss=1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight=np.array(weightGauss)*weight
    smoothed=[0.0]*(len(list)-window)
    for i in range(len(smoothed)):
        smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
    return smoothed

# Generate the polygon
theta = np.linspace(0,2*np.pi,200, endpoint=False)
r = np.random.lognormal(0,0.4,200)
r = np.pad(r,(9,10),mode='wrap')

r = smoothListGaussian(r, degree=10)

coords = zip(np.cos(theta)*r, np.sin(theta)*r)
coords.append(coords[0])
coords = np.array(coords)

# im = np.zeros([240,320],dtype=np.uint8)
# cv2.fillPoly( im, coords, 255 )
fig = plt.figure(1, figsize=(5,5), dpi=90)
ax = fig.add_subplot(111)
ax.plot(coords[:, 0], coords[:, 1], color='#6699cc', alpha=0.7,
        linewidth=3, solid_capstyle='round', zorder=2)
ax.set_title('Polygon')

plt.show()
