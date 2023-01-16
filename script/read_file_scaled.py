# script to read 'scaled_bucket_targeted_amount_xx_saved_trajs.pkl' files
#
# Z.Zhang
# 2023/1

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.spatial.transform import Rotation as R

expFolderName = 'C:/Users/zhangzeqing/OneDrive - connect.hku.hk/B_Data/From/Shiyu, Jin'
fileName = '/scaled_bucket_targeted_amount_0.6_saved_trajs.pkl' # unit: mm
save_path = expFolderName+'/fig'
f = open(expFolderName+fileName,'rb')
data = pickle.load(f)
# print(data)
# print(data['loader_type'])

paths_xyz = data['loader_pos_trajs']/1000 # unit: change to m
paths_theta = data['loader_rot_trajs']
num_path = np.shape(paths_xyz)[0]
num_traj = data['num_episodes']
path_length = data['episode_length']
bucket_front_length = data['bucket_front_length']
waterline_trajs = data['waterline_trajs'] # 5*75*1

# region
# for path_id in range(num_path):
#     fig = plt.figure(figsize = (10,10))
#     ax = plt.axes(projection='3d')

#     cur_path_xyz = paths_xyz[path_id,:,:]
#     # plot path
#     ax.plot3D(cur_path_xyz[:,0],cur_path_xyz[:,2],cur_path_xyz[:,1])
#     # start/goal
#     ax.scatter(cur_path_xyz[0,0],cur_path_xyz[0,2],cur_path_xyz[0,1], marker="o", color="red", s = 40,label="start")
#     ax.scatter(cur_path_xyz[-1,0],cur_path_xyz[-1,2],cur_path_xyz[-1,1], marker="x", color="red", s = 40,label="goal")


#     ax.set_title(f'{path_id}-th path')
#     # Set axes label
    # ax.axis('scaled')
    # ax.set_xlabel('x', labelpad=5)
    # ax.set_ylabel('y', labelpad=5)
    # ax.set_zlabel('z', labelpad=5)
    # ax.view_init(0 , 180)
#     plt.legend()
#     plt.show()
#     # plt.savefig(save_path+f'/{path_id}th_path' + '.png')
# endregion


## 2D plot in shiyu frame i.e., path in z-y plane
# region
# width = 0.05
# height = 0.025
# position_x = .1
# position_y = .4
# # extent = [position_x, position_x+width, position_y, position_y+height] # left, right, bottom, top
# init_color_matrix = [[1.,0.], [1.,0.]]

# for path_id in range(num_path):
#     fig = plt.figure(figsize = (10,10))
#     ax = plt.axes()
#     # ax = plt.axes(projection='3d')
#     # ax.view_init(0 , 180)

#     # path xyz R^{75*3}
#     cur_path_xyz = paths_xyz[path_id,:,:]
#     # theta (rad) R^{75*1}
#     rot_traj = paths_theta[path_id,:,:]

#     for i in range(path_length):
#         # in Shiyu's frame
#         # 3D plot path x,y,z
#         # ax.plot3D(cur_path_xyz[i,0],cur_path_xyz[i,2],cur_path_xyz[i,1],'.')
#         # 2D plot path y,z
#         ax.plot(cur_path_xyz[i,2],cur_path_xyz[i,1],'.')

#         # plot bucket model (rectangle)
#         position_x = cur_path_xyz[i,2]
#         position_y = cur_path_xyz[i,1]
#         angle_deg = np.degrees(rot_traj[i,0])
#         extent = [position_x, position_x+width, position_y-height/2, position_y+height/2] # left, right, bottom, top
#         car_img = ax.imshow(init_color_matrix, cmap = plt.cm.Greens, interpolation = 'bicubic', extent=extent)
#         trans_data = mtransforms.Affine2D().rotate_deg_around(position_x, position_y, angle_deg) + ax.transData
#         car_img.set_transform(trans_data)

#         # start/goal
#         ax.scatter(cur_path_xyz[0,2],cur_path_xyz[0,1], marker="o", color="red", s = 40,label="start")
#         ax.scatter(cur_path_xyz[-1,2],cur_path_xyz[-1,1], marker="x", color="red", s = 40,label="goal")


#         ax.axis('scaled')
#         ax.set_xlim([-0.2,0.3])
#         ax.set_ylim([0.1,0.7])
#         ax.set_xlabel('z', labelpad=5)
#         ax.set_ylabel('y', labelpad=5)
        
#         plt.show()
# endregion

## 3D plot in global(UR) frame i.e., path x-z plane
width = 0.12
height = 0.055
teeth_w = 0.16
teeth_h = 0.127
position_x = .1
position_y = .4
init_color_matrix = [[1.,0.], [1.,0.]]

### [bucket] oigin of shiyu frame expressed in global frame
oigin_pt = np.array([0.,0.,0.])

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
        ## plot point
        ax.plot3D(x_global,y_global,z_global,'.')

        ## quaternion in global frame
        theta_rad_global = -theta_rad_shiyu
        r = R.from_rotvec(theta_rad_global * np.array([0, 1, 0]))
        r_rotmat = R.from_euler('y', theta_rad_global).as_matrix()
        quat_global = r.as_quat()
        
        
        ## plot bucket outline
        ## (rectangle)
        # rect_pts_temp = np.array([[0,0,height/2],[width,0,height/2],[width,0,-height/2],[0,0,-height/2]]).T
        ## (5pts) 3*5
        # rect_pts_temp = np.array([[0,0,height/2],[width/2,0,height/2],[3*width/2,0,3*height/2],[width,0,-height/2],[0,0,-height/2]]).T
        rect_pts_temp = np.array([[0,0,height/2],[width/2,0,height/2],[teeth_w,0,teeth_h-height/2],[width,0,-height/2],[0,0,-height/2]]).T
        rect_pts_temp = np.hstack((rect_pts_temp,rect_pts_temp[:,0].reshape(3,-1)))
        # rect_pts_temp = np.array([width,0,0]).reshape(-1,1)
        # np.matmul(R,vect) + waypts[i,:].reshape(2,1)
        rect_pts_global = np.matmul(r_rotmat,rect_pts_temp) + np.array([x_global,y_global,z_global]).reshape(-1,1)
        xline = np.hstack((np.array(x_global), rect_pts_global[0,:]))
        yline = np.hstack((np.array(y_global), rect_pts_global[1,:]))
        zline = np.hstack((np.array(z_global), rect_pts_global[2,:]))
        

        ax.plot3D(xline,yline,zline,'-',color="darkgreen")

        ## start/goal
        # x_start_global = cur_path_xyz[0,2] + oigin_pt[0]
        # y_start_global = -cur_path_xyz[0,0]+ oigin_pt[1]
        # z_start_global = cur_path_xyz[0,1] + oigin_pt[2]
        # x_goal_global = cur_path_xyz[-1,2] + oigin_pt[0]
        # y_goal_global = -cur_path_xyz[-1,0]+ oigin_pt[1]
        # z_goal_global = cur_path_xyz[-1,1] + oigin_pt[2]
        # ax.scatter(x_start_global,y_start_global,z_start_global, marker="o", color="red", s = 40,label="start")
        # ax.scatter(x_goal_global,y_goal_global,z_goal_global, marker="x", color="red", s = 40,label="goal")

        # ax.axis('scaled')
        # ax.set_xlim([-0.2,0.3])
        # ax.set_zlim([0.1,0.7])
        # ax.set_xlabel('x', labelpad=5)
        # ax.set_ylabel('y', labelpad=5)
        # ax.set_zlabel('z', labelpad=5)        

    
    # start/goal
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