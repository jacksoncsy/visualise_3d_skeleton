"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((32, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))
        

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750    
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')
    
    
""" 17-point pose
0: 'pelvis', 1: 'right_hip', 2: 'right_knee', 3: 'right_foot', 4: 'left_hip', 5: 'left_knee', 6: 'left_foot', 
7: 'body_center', 8: 'neck', 9: 'nose', 10: 'head',
11: 'left_shoulder', 12: 'left_arm', 13: 'left_hand', 14: 'right_shoulder', 15: 'right_arm', 16: 'right_hand'
"""
##### links w/ neck and pelvis #####
JOINT_LINKS = [
    [10, 9], [9, 8], [8, 7], [7, 0],
    [8, 14], [14, 15], [15, 16], 
    [8, 11], [11, 12], [12, 13],
    [0, 1], [1, 2], [2, 3], 
    [0, 4], [4, 5], [5, 6]         
]


class Ax3DPose17Point(object):
  def __init__(self, ax, r=1, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """        
    self.joints = JOINT_LINKS
    self.num_points = 17
    # Left / right indicator        
    self.LR  = np.array([1,1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1], dtype=bool)
    
    self.I = np.array(self.joints)[:,0]
    self.J = np.array(self.joints)[:,1]
    
    self.ax = ax 
    self.r = r    

    # Make connection matrix
    self.plots = []
    self.vals = np.zeros((self.num_points, 3))
    for i in np.arange( len(self.I) ):
      x = np.array( [self.vals[self.I[i], 0], self.vals[self.J[i], 0]] )
      y = np.array( [self.vals[self.I[i], 1], self.vals[self.J[i], 1]] )
      z = np.array( [self.vals[self.I[i], 2], self.vals[self.J[i], 2]] )      
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))      
        

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 17-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.shape == (17,3), "channels should be (17,3)"    
        
    self.vals = channels

    for i in np.arange( len(self.I) ):
      x = np.array( [self.vals[self.I[i], 0], self.vals[self.J[i], 0]] )
      y = np.array( [self.vals[self.I[i], 1], self.vals[self.J[i], 1]] )
      z = np.array( [self.vals[self.I[i], 2], self.vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = self.r
    xroot, yroot, zroot = self.vals[0,0], self.vals[0,1], self.vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r*r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')
