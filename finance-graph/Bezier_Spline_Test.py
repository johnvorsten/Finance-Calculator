# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:13:29 2019

@author: z003vrzk
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from datetime import date, timedelta, datetime
from math import factorial



time_start = date(2016, 9, 28)
time_end = date(2019, 6, 5)

delta = time_end - time_start
dates = [(time_start + timedelta(days=i)) for i in range(delta.days)]
m = delta.days
n = 11
_path = r".\JVImports\Finance_Graph_values.npy"
values = np.load(_path)

"""New Method"""
#1 use scipy scipy.interpolate.bisplev with scipy.interpolate.bisplrep
#2 OR calculate bezier surface directly


#Get vals to a points matrix of rank 2 [[x,y,z], [...]]
m, n = values.shape[0], values.shape[1]
pts = np.zeros((m*n, 3))
idx = 0
for _row in range(m):
    for _col in range(n):
        pts[idx, 0] = _row
        pts[idx, 1] = _col
        pts[idx, 2] = values[_row,_col]
        idx += 1

sets = 20
_segment = min(sets*n, pts.shape[0]*0.1)
tck = interpolate.bisplrep(pts[:_segment,0], pts[:_segment,1], pts[:_segment,2])

x_grid = np.arange(0, values.shape[0], 1)
y_grid = np.linspace(1, values.shape[1],num=11)
vals = interpolate.bisplev(x_grid, y_grid, tck)


"""Plotting method"""
#Start doing plotting
fig3d = plt.figure(1)
#fig2d = plt.figure(2)
ax3d = fig3d.add_subplot(111, projection='3d')
#ax2d = fig2d.add_subplot(111)

smooth = True
if smooth:
    m, n = values.shape[0], values.shape[1]
    pts = np.zeros((m*n, 3))
    idx = 0
    
    for _row in range(m):
        for _col in range(n):
            pts[idx, 0] = _row
            pts[idx, 1] = _col
            pts[idx, 2] = values[_row,_col]
            idx += 1
            
#    sets = 100
#    _segment = int(min(sets*n, pts.shape[0]*0.5))
#    tck = interpolate.bisplrep(pts[:_segment,0], pts[:_segment,1], pts[:_segment,2])
    tck = interpolate.bisplrep(pts[:,0], pts[:,1], pts[:,2])
    
    x_grid = np.arange(0, values.shape[0], 1)
    y_grid = np.linspace(1, values.shape[1],num=11)
    Z3d = interpolate.bisplev(x_grid, y_grid, tck).transpose()
    
    X_ = np.arange(0, len(dates), 1)
    y_ = np.linspace(1, 0, num=11)
    X, y = np.meshgrid(X_, y_)
    
else:
    X_ = np.arange(0,len(dates), 1)
    y_ = np.linspace(1,0,num=11)
    X, y = np.meshgrid(X_, y_)
    Z3d = values.transpose() #Get integrated values

#Z2d = values.mean(axis=1)



surf = ax3d.plot_surface(X, y, Z3d, cmap='summer', linewidth=0, 
                       antialiased=False)
#line = ax2d.plot(X_, Z2d, label='Bank Value', linewidth=2)

num_ticks = 4
tick_index = np.linspace(min(X_), max(X_), num=num_ticks, dtype=np.int16)
#ticks = X[tick_index]
ticks = X_[tick_index]
ax3d.set_xticks(ticks) #Set the locations of tick marks from sequence ticks
#ax2d.set_xticks(ticks)

label_index = tick_index
label_date = [dates[idx] for idx in label_index]
labels = [x.isoformat() for x in label_date]
ax3d.set_xticklabels(labels) #Define the strings to be defined
#ax2d.set_xticklabels(labels)
fig3d.colorbar(surf, shrink=0.5, aspect=5)

ax3d.set_xlabel('Date')
ax3d.set_ylabel('Probability')
ax3d.set_zlabel('Total Value')
#ax2d.set_xlabel('Date')
#ax2d.set_ylabel('Total Value')

ax3d.view_init(elev=20, azim=125) #elev is in z plane, azim is in x,y plane

for angle in range(100,125):
    ax3d.view_init(elev=20, azim=angle)
    plt.draw()
    plt.pause(0.003)

plt.show()
















def bernstein_poly(i, n, t):
    return (factorial(n)/(factorial(i)*factorial(n-i))) * t**i * (1 - t)**(n - i)










































#"""Old plotting method"""
##Start doing plotting
#fig3d = plt.figure(1)
#fig2d = plt.figure(2)
#ax3d = fig3d.add_subplot(111, projection='3d')
#ax2d = fig2d.add_subplot(111)
#
#X_ = np.arange(0,len(dates), 1)
#y_ = np.linspace(1,0,num=11)
#X, y = np.meshgrid(X_, y_)
#Z3d = values.transpose() #Get integrated values
#Z2d = values.mean(axis=1)
#
#
#surf = ax3d.plot_surface(X, y, Z3d, cmap='summer', linewidth=0, 
#                       antialiased=False)
#line = ax2d.plot(X_, Z2d, label='Bank Value', linewidth=2)
#
#num_ticks = 4
#tick_index = np.linspace(min(X_), max(X_), num=num_ticks, dtype=np.int16)
##ticks = X[tick_index]
#ticks = X_[tick_index]
#ax3d.set_xticks(ticks) #Set the locations of tick marks from sequence ticks
#ax2d.set_xticks(ticks)
#
#label_index = tick_index
#label_date = [dates[idx] for idx in label_index]
#labels = [x.isoformat() for x in label_date]
#ax3d.set_xticklabels(labels) #Define the strings to be defined
#ax2d.set_xticklabels(labels)
#fig3d.colorbar(surf, shrink=0.5, aspect=5)
#
#ax3d.set_xlabel('Date')
#ax3d.set_ylabel('Probability')
#ax3d.set_zlabel('Total Value')
#ax2d.set_xlabel('Date')
#ax2d.set_ylabel('Total Value')
#
#ax3d.view_init(elev=20, azim=125) #elev is in z plane, azim is in x,y plane
#
#for angle in range(100,125):
#    ax3d.view_init(elev=20, azim=angle)
#    plt.draw()
#    plt.pause(0.003)
#
#plt.show()