from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import copy
import scipy.io

def main():
    xyz_gt = np.load('xyz_gt.npy') # xyz,joint,frame    
    frame_number = xyz_gt.shape[0]
    
    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax)

    # Plot the conditioning ground truth
    for i in range(frame_number):
        ob.update(xyz_gt[i, :])
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    main()
