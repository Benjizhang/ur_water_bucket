
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

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



width = 0.05
height = 0.025
position_x = .1
position_y = .4
# extent = [position_x, position_x+width, position_y, position_y+height] # left, right, bottom, top
init_color_matrix = [[1.,0.], [1.,0.]]

for path_id in range(num_path):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes()
    # ax = plt.axes(projection='3d')
    # ax.view_init(0 , 180)

    # path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    # theta (rad) R^{75*1}
    rot_traj = paths_theta[path_id,:,:]

    for i in range(path_length):
        # in Shiyu's frame
        # 3D plot path x,y,z
        # ax.plot3D(cur_path_xyz[i,0],cur_path_xyz[i,2],cur_path_xyz[i,1],'.')
        # 2D plot path y,z
        ax.plot(cur_path_xyz[i,2],cur_path_xyz[i,1],'.')

        # plot bucket model (rectangle)
        position_x = cur_path_xyz[i,2]
        position_y = cur_path_xyz[i,1]
        angle_deg = np.degrees(rot_traj[i,0])
        extent = [position_x, position_x+width, position_y-height/2, position_y+height/2] # left, right, bottom, top
        car_img = ax.imshow(init_color_matrix, cmap = plt.cm.Greens, interpolation = 'bicubic', extent=extent)
        trans_data = mtransforms.Affine2D().rotate_deg_around(position_x, position_y, angle_deg) + ax.transData
        car_img.set_transform(trans_data)

        # start/goal
        ax.scatter(cur_path_xyz[0,2],cur_path_xyz[0,1], marker="o", color="red", s = 40,label="start")
        ax.scatter(cur_path_xyz[-1,2],cur_path_xyz[-1,1], marker="x", color="red", s = 40,label="goal")


        ax.axis('scaled')
        ax.set_xlim([-0.2,0.3])
        ax.set_ylim([0.1,0.7])
        ax.set_xlabel('z', labelpad=5)
        ax.set_ylabel('y', labelpad=5)
        
