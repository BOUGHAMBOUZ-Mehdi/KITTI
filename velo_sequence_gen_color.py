import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import plyfile as ply
basedir, date, drive = '../../kitti_dataset/RAW', '2011_09_26', '0009'
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 50, 1))
rng = range(0, 50)
min_dist = 5
pc_xyz  = None
pc_rgb = None
for i in rng:
    img = dataset.get_cam2(i)
    w, h = img.size
    velo = dataset.get_velo(i)
    velo  = velo[velo[:,0] > min_dist,:]
    velo[:, 3] = 1    
    K_cam2 = np.eye(4)
    K_cam2[:3,:3] = dataset.calib.K_cam2
    P_velo_to_img = K_cam2.dot(dataset.calib.T_cam2_velo)
    proj_2D = P_velo_to_img.dot(velo.transpose())
    proj_2D = proj_2D[0:3, :]
    proj_2D[0, :] = proj_2D[0, :] / proj_2D[2, :]
    proj_2D[1, :] = proj_2D[1, :] / proj_2D[2, :]
    proj_2D = proj_2D[0:2, :].transpose()
    cond1 = np.logical_or(proj_2D[:,0] > w, proj_2D[:,1] >  h)
    cond2 = np.logical_or(proj_2D[:,0] < 0, proj_2D[:,1] < 0)
    valid_idx = np.invert(np.logical_or(cond1,cond2))
    proj_2D = proj_2D[valid_idx,:]
    velo = velo[valid_idx,:]
    proj_2D = np.floor(proj_2D)
    color_rgb = [img.getpixel(tuple(proj_2D[i,:])) for i in range(len(proj_2D))]
    if pc_xyz is None:
        pc_xyz = dataset.oxts[i].T_w_imu.dot(np.linalg.inv(dataset.calib.T_velo_imu).dot(velo.transpose()))
        pc_rgb = color_rgb
    else:
        velo_trans = dataset.oxts[i].T_w_imu.dot(np.linalg.inv(dataset.calib.T_velo_imu).dot(velo.transpose()))
        pc_xyz = np.concatenate((pc_xyz, velo_trans), axis=1)
        pc_rgb = pc_rgb  + color_rgb

pc_xyz_tuples = [(i,j,k) for i,j,k in tuple(pc_xyz[0:3,:].transpose())]
pc_xyzrgb = [pc_xyz_tuples[i]+pc_rgb[i][:] for i in range(len(pc_rgb))]
vertex = np.array(pc_xyzrgb,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red','u1'), ('green','u1'), ('blue','u1')])
el = ply.PlyElement.describe(vertex, 'vertex')
ply.PlyData([el]).write('binary.ply')