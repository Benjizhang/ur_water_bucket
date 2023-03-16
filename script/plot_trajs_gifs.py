# Z.Zhang
# 2023/2
# Revised by Yaru Niu, 2023/3

import pickle,glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.spatial.transform import Rotation as R
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
import os

# amount_goal_ls = [0.6, 0.7]
# pos_goal_ls = [2]
# waterline_ls = [2]
# start_ite, end_ite, setp_ite = 0, 110, 5
# folder_path = 'paper_trajs'
# traj_folder_path = '/waterlines_7_8_9/scaled_trajs_23cm'

amount_goal_ls = [0.6]
pos_goal_ls = [2]
waterline_ls = [3]
start_ite, end_ite, setp_ite = 0, 110, 5
folder_path = 'paper_trajs'
traj_folder_path = '/waterlines_6_6.5_7.5/scaled_trajs_23cm'

# amount_goal_ls = [0.6]
# pos_goal_ls = [2]
# waterline_ls = [2]
# start_ite, end_ite, setp_ite = 0, 110, 5
# folder_path = 'paper_trajs'
# traj_folder_path = '/waterlines_7_8.5_10/scaled_trajs_23cm'

# amount_goal_ls = [0.65]
# pos_goal_ls = [1, 2, 3]
# waterline_ls = [2]
# start_ite, end_ite, setp_ite = 0, 110, 5
# folder_path = 'paper_trajs'
# traj_folder_path = '/waterlines_7_8.5_10/scaled_trajs_30cm'

# amount_goal_ls = [0.65]
# pos_goal_ls = [1, 2, 3]
# waterline_ls = [2]
# start_ite, end_ite, setp_ite = 0, 110, 5
# folder_path = 'paper_trajs'
# traj_folder_path = '/waterlines_7_8.5_10/scaled_trajs_40cm'


saveFig = 1
NutStorePath = 'D:/Niu, Yaru/23cm_7_8_9cm'
trajFolderName = '/scaled_trajs_035'
savePath = NutStorePath+'/fig_23cm_7_8_9cm'

# save_path = folder_path + traj_folder_path + '_plots/'
save_path = savePath + '/plots'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for ii in range(len(amount_goal_ls)):
    for jj in range(len(pos_goal_ls)):
        for kk in range(len(waterline_ls)):
            amount_goal = amount_goal_ls[ii]
            pos_goal    = pos_goal_ls[jj]
            waterline   = waterline_ls[kk]
            
            print(f'------ amount_goal: {amount_goal} ------')
            print(f'------ pos_goal:    {pos_goal} ------')
            print(f'------ waterline:   {waterline} ------')
            
            file_name = '/bucket_amount_goal_'+str(amount_goal)+'_pos_goal_'+str(pos_goal)+'_waterline_'+str(waterline)+'_seed_0_error_*.pkl' # unit: mm
            # file_dir = folder_path + traj_folder_path +file_name
            file_dir = NutStorePath+trajFolderName+file_name

            if len(glob.glob(file_dir)) != 1:
                raise Exception('Multiple pkl files!')
            file = glob.glob(file_dir)[0]
            print('---- {} ----'.format(file))
            f = open(file,'rb')
            data = pickle.load(f)

            paths_xyz = data['loader_pos_trajs']/1000 # unit: change to m
            paths_theta = data['loader_rot_trajs']
            num_path = np.shape(paths_xyz)[0]
            num_traj = data['num_episodes']
            path_length = data['episode_length']
            bucket_front_length = data['bucket_front_length']
            waterline_trajs = data['waterline_trajs'] # 1*110*1
            in_loader_percent_trajs = data['in_loader_percent_trajs'] # 1*110*1
            desired_pos_goal = data['targeted_pos_trajs'][0, 0] / 1000

            ## 3D plot in global(UR) frame i.e., path x-z plane
            width = 0.12
            height = 0.055
            teeth_w = 0.16
            teeth_h = 0.127

            ### [bucket] oigin of shiyu frame expressed in global frame
            oigin_pt = np.array([0.,0.,0.])
            plot_waterline = 0


            path_id = 0
            # path xyz R^{110*3}
            cur_path_xyz = paths_xyz[path_id,:,:]
            # theta (rad) R^{110*1}
            rot_traj = paths_theta[path_id,:,:]
            # water line height
            waterline_h = waterline_trajs[path_id,:,:]/1000
            first_waterline_h = waterline_h[0,0]
            mean_waterline_h = np.mean(waterline_h)
            print(f'--*-- path {path_id} --*--')
            print(f'first waterline_h: {first_waterline_h}')
            print(f'mean of waterline_h: {mean_waterline_h}')
            if plot_waterline == 1:
                fig2 = plt.figure(figsize = (10,10))
                ax2 = plt.axes()
                ax2.plot(waterline_h)
                ax2.set_xlabel('time_step', labelpad=5)
                ax2.set_ylabel('height (m)', labelpad=5) 
                ax2.set_ylim([0.,0.3])
                plt.grid()
                plt.show()
            

            fig = plt.figure(figsize = (10,10))
            fig.subplots_adjust(left=0., bottom=0., right=0.99, top=0.99, wspace=0., hspace=0)
            ax = plt.axes(projection='3d')
            # ax.view_init(17,82)
            ax.view_init(0,90)
            
            def animate(i):
                ## in Shiyu's frame
                # print(cur_path_xyz.shape)
                x_shiyu = cur_path_xyz[i,0]
                y_shiyu = cur_path_xyz[i,1]
                z_shiyu = cur_path_xyz[i,2]
                theta_deg_shiyu = np.degrees(rot_traj[i,0])
                theta_rad_shiyu = rot_traj[i,0]

                ## convert shiyu frame to UR frame (global)
                x_global = z_shiyu + oigin_pt[0]
                y_global = -x_shiyu+ oigin_pt[1]
                z_global = y_shiyu + oigin_pt[2]

                ## quaternion in global frame
                theta_rad_global = -theta_rad_shiyu
                r = R.from_rotvec(theta_rad_global * np.array([0, 1, 0]))
                r_rotmat = R.from_euler('y', theta_rad_global).as_matrix()
                quat_global = r.as_quat()

                ## cal. xyz of EE
                # xyz_EE_global = addFT(x_global,y_global,z_global,r)
                # wpose.position.x = xyz_EE_global[0]
                # wpose.position.y = xyz_EE_global[1]
                # wpose.position.z = xyz_EE_global[2]
                # ax.plot3D(xyz_EE_global[0],xyz_EE_global[1],xyz_EE_global[2],'b.')
                
                
                ## plot bucket outline
                ## (rectangle)
                # rect_pts_temp = np.array([[0,0,height/2],[width,0,height/2],[width,0,-height/2],[0,0,-height/2]]).T
                ## (5pts) 3*5
                rect_pts_temp = np.array([[0,0,height/2],[width/2,0,height/2],[teeth_w,0,teeth_h-height/2],[width,0,-height/2],[0,0,-height/2]]).T
                rect_pts_temp = np.hstack((rect_pts_temp,rect_pts_temp[:,0].reshape(3,-1)))
                rect_pts_global = np.matmul(r_rotmat,rect_pts_temp) + np.array([x_global,y_global,z_global]).reshape(-1,1)
                xline = np.hstack((np.array(x_global), rect_pts_global[0,:]))
                yline = np.hstack((np.array(y_global), rect_pts_global[1,:]))
                zline = np.hstack((np.array(z_global), rect_pts_global[2,:]))
                ## draw outline of bucket
                # ax.plot3D(xline,yline,zline,'-',color="darkgreen")
                ## bucket_patch
                Path = mpath.Path
                path_data = [ 
                (Path.MOVETO, [xline[0],zline[0]]),
                (Path.LINETO, [xline[1],zline[1]]),
                (Path.LINETO, [xline[2],zline[2]]),
                (Path.LINETO, [xline[3],zline[3]]),
                (Path.LINETO, [xline[4],zline[4]]),
                (Path.LINETO, [xline[5],zline[5]]),   
                (Path.CLOSEPOLY, [x_global,z_global]),
                ]
                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)
                bucket_patch = mpatches.PathPatch(path, facecolor='green', alpha=0.18)
                ax.add_patch(bucket_patch)
                art3d.pathpatch_2d_to_3d(bucket_patch, z=0, zdir="y")  
                
                # start/goal
                x_start_global = cur_path_xyz[0,2] + oigin_pt[0]
                y_start_global = -cur_path_xyz[0,0]+ oigin_pt[1]
                z_start_global = cur_path_xyz[0,1] + oigin_pt[2]
                x_final_global = cur_path_xyz[-1,2] + oigin_pt[0]
                y_final_global = -cur_path_xyz[-1,0]+ oigin_pt[1]
                z_final_global = cur_path_xyz[-1,1] + oigin_pt[2]
                # plot water line
                x_waterline = np.array([-0.2,0.39])
                y_waterline = np.array([0,0])
                z_waterline = np.array([first_waterline_h,first_waterline_h])
                
                if len(ax.get_legend_handles_labels()[1]) == 0:
                    ax.scatter(x_start_global,y_start_global,z_start_global, marker="o", color="red", s = 40, label="initial position")
                    # ax.scatter(x_final_global,y_final_global,z_final_global, marker="x", color="red", s = 60,label="final position")
                    ax.scatter(desired_pos_goal[2], desired_pos_goal[0], desired_pos_goal[1], marker="x", color="red", s = 60, label="desired position goal")
                    ax.plot3D(x_waterline,y_waterline,z_waterline,color="blue",label = 'initial waterline')
                    ## plot point
                    # ax.plot3D(x_global,y_global,z_global,'.')
                    ax.scatter(x_global, y_global, z_global, c = 'firebrick', s = 10, label='trajectory')
                else:
                    ax.scatter(x_start_global,y_start_global,z_start_global, marker="o", color="red", s = 40)
                    # ax.scatter(x_final_global,y_final_global,z_final_global, marker="x", color="red", s = 60,label="final position")
                    ax.scatter(desired_pos_goal[2], desired_pos_goal[0], desired_pos_goal[1], marker="x", color="red", s = 60)
                    # ax.plot3D(x_waterline,y_waterline,z_waterline,color="blue")
                    ax.scatter(x_global, y_global, z_global, c = 'firebrick', s = 10)
                
                # figure setting
                ax.axis('scaled')
                ax.set_box_aspect((1, 1, 1))
                ax.set_xlim([-0.20,0.39])
                ax.set_zlim([0.,0.587])
                # ax.axis('scaled')

                plt.tick_params(axis = 'x',labelsize=15)
                plt.tick_params(axis = 'z',labelsize=15)
                ax.set_xlabel('x', labelpad=15, fontsize=20)
                ax.set_ylabel('y', labelpad=15, fontsize=20)
                ax.set_zlabel('z', labelpad=20, fontsize=20)   
                
                    # ax.set_title(f'AmountGoal {amount_goal}, PosGoal {pos_goal}, WaterLine {waterline}')
                plt.legend(loc=(0.55, 0.67), fontsize=12)
                    
                
            ani = animation.FuncAnimation(fig, animate, repeat=True, frames=110, interval=200)
            
            # plt.show()
            
            fig_name = f'AG{amount_goal}_PG{pos_goal}_WL{waterline}.gif'
            ani.save(save_path+'/' + fig_name, writer='ffmpeg', fps=30)
