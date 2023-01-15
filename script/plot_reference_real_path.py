## read given path (reference) and recorded path (real)
## plot in one figure
#
# Z. Zhang
# 2023/1

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.spatial.transform import Rotation as R

import pandas as pd
import os, os.path, glob

expFolderName = 'C:/Users/zhangzeqing/OneDrive - connect.hku.hk/B_Data/From/Shiyu, Jin'
fileName = '/bucket_targeted_amount_0.6_saved_trajs.pkl'
save_path = expFolderName+'/fig'
f = open(expFolderName+fileName,'rb')
data = pickle.load(f)
# print(data)
# print(data['loader_type'])

paths_xyz = data['loader_pos_trajs']
paths_theta = data['loader_rot_trajs']
num_path = np.shape(paths_xyz)[0]
num_traj = data['num_episodes']
path_length = data['episode_length']

## [bucket] size
width = 0.05
height = 0.025
position_x = .1
position_y = .4
## [bucket] oigin of shiyu frame expressed in global frame
oigin_pt = np.array([0.,0.,0.])

ref_x_global,ref_y_global,ref_z_global =[],[],[]

######### log #########
## log path
expFolderName = '/20230115_exp' # <<<<<<
NutStorePath = 'D:/02data/MyNutFiles/我的坚果云/water_manipulate'
dataPath = NutStorePath+expFolderName+'/data'
figPath = NutStorePath+expFolderName+'/fig'
os.chdir(dataPath)

slide_id_ls = np.arange(num_path)
fd_data_ls = [None] * len(slide_id_ls)

for path_id in range(num_path):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # ax.view_init(17,82)
    ax.view_init(0,-90)

    # path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    # theta (rad) R^{75*1}
    rot_traj = paths_theta[path_id,:,:]

    for i in range(37):
        ## in Shiyu's frame
        x_shiyu = cur_path_xyz[i,0]
        y_shiyu = cur_path_xyz[i,1]
        z_shiyu = cur_path_xyz[i,2]
        theta_deg_shiyu = np.degrees(rot_traj[i,0])
        theta_rad_shiyu = rot_traj[i,0]

        ## convert shiyu frame to UR frame (global)
        x_global = z_shiyu + oigin_pt[0]
        y_global = -x_shiyu+ oigin_pt[1]
        z_global = y_shiyu + oigin_pt[2]

        ref_x_global.append(x_global)
        ref_y_global.append(y_global)
        ref_z_global.append(z_global)
    
    ## plot points of reference path
    ax.plot3D(ref_x_global,ref_y_global,ref_z_global,'r.',markersize=5,label='reference')
        
    ## plot real path
    file_suffix = '*path'+str(path_id)+'_rela_xz.csv'
    for file in glob.glob(file_suffix):
        print('---- {} ----'.format(file))
        fd_data_ls[path_id] = pd.read_csv(file,names=['relax','relaz'])
        real_x_global = fd_data_ls[path_id].relax
        real_y_global = np.zeros(np.shape(real_x_global))
        real_z_global = fd_data_ls[path_id].relaz        
        ## plot real path
        ax.plot3D(real_x_global,real_y_global,real_z_global,'b-',label='real')


    ## start/goal
    x_start_global = cur_path_xyz[0,2] + oigin_pt[0]
    y_start_global = -cur_path_xyz[0,0]+ oigin_pt[1]
    z_start_global = cur_path_xyz[0,1] + oigin_pt[2]
    x_goal_global = cur_path_xyz[-1,2] + oigin_pt[0]
    y_goal_global = -cur_path_xyz[-1,0]+ oigin_pt[1]
    z_goal_global = cur_path_xyz[-1,1] + oigin_pt[2]
    ax.scatter(x_start_global,y_start_global,z_start_global, marker="o", color="red", s = 40,label="start")
    ax.scatter(x_goal_global,y_goal_global,z_goal_global, marker="x", color="red", s = 40,label="goal")

    ax.axis('scaled')
    ax.set_xlim([-0.2,0.3])
    ax.set_zlim([0.1,0.7])
    ax.axis('scaled')
    ax.set_xlabel('x', labelpad=5)
    ax.set_ylabel('y', labelpad=5)
    ax.set_zlabel('z', labelpad=5)   
    ax.set_title(f'{path_id}-th path')
    plt.legend()
    plt.show()
