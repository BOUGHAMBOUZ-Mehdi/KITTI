import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti

basedir = '../../kitti_dataset/RAW'
date = '2011_09_26'
drive = '0009'
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))
frame, min_dist = 0,5
img = dataset.get_cam2(frame)
w, h = img.size
velo = dataset.get_velo(frame)
velo  = velo[velo[:,0] > min_dist,:]
K_cam2 = np.eye(4)
K_cam2[:3,:3] = dataset.calib.K_cam2
P_velo_to_img = K_cam2.dot(dataset.calib.T_cam2_velo)
velo[:,3]=1
proj_2D = P_velo_to_img.dot(velo.transpose())
proj_2D  = proj_2D[0:3, :]
proj_2D[0, :] = proj_2D[0, :] / proj_2D[2, :]
proj_2D[1, :] = proj_2D[1, :] / proj_2D[2, :]
proj_2D = proj_2D[0:2, :].transpose()
cond1 = np.logical_or(proj_2D[:,0] > w, proj_2D[:,1] >  h)
cond2 = np.logical_or(proj_2D[:,0] < 0, proj_2D[:,1] < 0)
valid_idx = np.invert(np.logical_or(cond1,cond2))
proj_2D  = proj_2D[valid_idx,:]
velo  = velo[valid_idx,:]
color = 64*5./velo[:,0]
plt.imshow(img)
plt.scatter(proj_2D[:,0],proj_2D[:,1],s=5,c=color,cmap='jet')
plt.axis('off')
plt.show()