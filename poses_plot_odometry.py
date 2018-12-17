import matplotlib.pyplot as plt
import numpy as np

import pykitti

basedir = '../../../kitti_dataset/ODOMETRY_TP/dataset'
sequence = '04'
rng = range(0, 270, 1)
dataset = pykitti.odometry(basedir, sequence, frames=rng)
np.set_printoptions(precision=4, suppress=True)

for i in rng:
    plt.scatter(dataset.poses[i][2,3],dataset.poses[i][0,3],s= 5,c='r')

plt.axis('equal')

plt.show()
