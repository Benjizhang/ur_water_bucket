# exp: ur5 + path data from the folder 'data'
# for scaled data
# _hk: exp for [0.6,0.7,0.8,0.65,0.75] pkl files
#      using 327Box, 327 table
# _left_dump: new postion to dump and scale water
#
# Z. Zhang
# 2/23

from cmath import cos
import copy
import time
import rospy
import numpy as np
from functions.bagRecorder import BagRecorder
from functions.scene_helper import zero_ft_sensor,ft_listener
from functions.ur_move import MoveGroupPythonInteface
import moveit_commander
import sys,csv,glob,pickle
from scipy.spatial.transform import Rotation as R

# to read 'scaled_xxx' files
# unit: change to m
# considering delta offset from FT300 sensor (0.0375m)
def urGivenPath4(ur_control,file_dir,path_id,oigin_pt,oigin_angle_rad,ite_start,ite_end):
    ## ur
    waypoints1 = []
    wpose = ur_control.group.get_current_pose().pose

    ## given path info
    if len(glob.glob(file_dir)) != 1:
        raise Exception('Multiple pkl files!')
    
    file = glob.glob(file_dir)[0]
    print('---- {} ----'.format(file))
    f = open(file,'rb')
    data = pickle.load(f)
    paths_xyz = data['loader_pos_trajs']/1000 # unit: change to m
    paths_theta = data['loader_rot_trajs']
    num_path = np.shape(paths_xyz)[0]
    path_length = data['episode_length']

    if path_id >= num_path:
        raise Exception('path id error')

    ## path xyz R^{110*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    ## theta (rad) R^{110*1}
    rot_traj = paths_theta[path_id,:,:]

    for i in range(ite_start,ite_end):
        ## position in Shiyu's frame
        x_shiyu = cur_path_xyz[i,0]
        y_shiyu = cur_path_xyz[i,1]
        z_shiyu = cur_path_xyz[i,2]
        theta_deg_shiyu = np.degrees(rot_traj[i,0])
        theta_rad_shiyu = rot_traj[i,0]

        ## convert shiyu frame to UR frame (global)
        x_global = z_shiyu + oigin_pt[0]
        y_global = -x_shiyu+ oigin_pt[1]
        z_global = y_shiyu + oigin_pt[2]

        # wpose.position.x = x_global
        # wpose.position.y = y_global
        # wpose.position.z = z_global        

        ## quaternion in global frame
        theta_rad_global = -theta_rad_shiyu
        r = R.from_euler('y', theta_rad_global+oigin_angle_rad)
        r_rela = R.from_euler('y', theta_rad_global)
        ## cal. xyz of EE
        xyz_EE_global = addFT(x_global,y_global,z_global,r_rela)
        wpose.position.x = xyz_EE_global[0]
        wpose.position.y = xyz_EE_global[1]
        wpose.position.z = xyz_EE_global[2]

        # r_rotmat = R.from_euler('y', theta_rad_global).as_matrix()
        quat_global = r.as_quat()
        
        wpose.orientation.x = quat_global[0]
        wpose.orientation.y = quat_global[1]
        wpose.orientation.z = quat_global[2]
        wpose.orientation.w = quat_global[3]

        waypoints1.append(copy.deepcopy(wpose))
    return waypoints1

# frame transfer when adding a FT300
# xyz is center of FT300, to cal. xyz of EE
def addFT(x,y,z,r,delta_L=0.0375):
    rot_mat = r.as_matrix()
    xyz_FT = np.array([x,y,z]).reshape(-1,1)
    delta_vec = np.array([-delta_L,0,0]).reshape(-1,1)
    xyz_EE = np.matmul(rot_mat,delta_vec)+xyz_FT

    return xyz_EE.flatten()



if __name__ == '__main__':
    rospy.init_node("test_move")
    moveit_commander.roscpp_initialize(sys.argv)

    ## input the exp mode
    exp_mode = 'trial'
    print('**** Exp Mode: '+exp_mode+' ****')
    rospy.sleep(0.5)

    ############# ur control #############
    # ur_control = MoveGroupPythonInteface(sim=True)  #simu
    ur_control = MoveGroupPythonInteface(sim=False)  #real
    rospy.loginfo('init ok.')

    ur_control.group.get_planning_frame()
    ur_control.group.get_end_effector_link()
    ur_control.remove_allobjects()
    res = ur_control.play_program()
    rospy.loginfo("play_program: {}".format(res))
    rospy.sleep(1)    
    
    LIFT_HEIGHT = +0.10 #(default: +0.10) # <<<<<<   
    # depthz = originz + PENE_DEPTH
    maxVelScale    = 0.6 # <<<<<<
    # Cur SAFE FORCE
    CUR_SAFE_FORCE = 45.0  #(default: 15N) # <<<<<<
    
    # folder name
    expFolderName = '/20230225_exphk' # <<<<<<
    NutStorePath = '/home/ur5/Nutstore Files/Nutstore/water_manipulate'
    DataPKLPath = '/home/ur5/Niu, Yaru/40cm_7_8_9cm'
    dataPath = NutStorePath+expFolderName+'/data'
    figPath = NutStorePath+expFolderName+'/fig'
    bagPath = '/home/ur5/ur5_ws/src/ur_control/logs' #'/home/ur5/rosbag'
    isSaveForce = 1           # <<<<<<
    isRecord = 1 # rosbag recorder

    ## set zero to the ft 300
    zero_ft_sensor()
    rospy.sleep(0.5)
    listener = ft_listener()

    ## velocity setting
    ur_control.set_speed_slider(maxVelScale)
    
    ## horizonal angle
    hori_angle_rad = -4.251274840994698

    amount_goal_ls = [0.6,0.7,0.8,0.65,0.75]

    amount_goal = 0.7
    pos_goal    = 3
    waterline   = 2 #<<<< serious when change it
    print(f'------ amount_goal: {amount_goal} ------')
    print(f'------ pos_goal:    {pos_goal} ------')
    print(f'------ waterline:   {waterline} ------')

    ## [bucket] start point x=0,y=0.4,z=0 expressed in shiyu frame 
    start_pt = [-0.54943859455702817, 0.10205824512513274,0.08795793366304825]
    start_pt[0] -= 0.1
    start_pt[1] += 0.05
    start_pt[2] += 0.056+0.1#-0.067

    # 327Box left dump
    x_pos_scale = 0.05280974577389467
    y_pos_scale = 0.5545436377897706
    z_pos_scale = start_pt[2]-0.1#+0.067
    # [scale] to scale bucket filling rate or not
    needScale = 1

    waypoints = []
    wpose = ur_control.group.get_current_pose().pose
    ## [bucket] oigin of shiyu frame expressed in global frame
    oigin_pt = np.array(start_pt)-np.array([0,0,0.4])

    ## [bucket] give start angle of bucket at start pt
    oigin_angle_rad = -1.5*np.pi
    r = R.from_euler('y', 0+oigin_angle_rad)
    quat_global = r.as_quat()
    wpose.orientation.x = quat_global[0]
    wpose.orientation.y = quat_global[1]
    wpose.orientation.z = quat_global[2]
    wpose.orientation.w = quat_global[3]

    r_rela = R.from_euler('y', 0)
    start_pt_EE = addFT(start_pt[0],start_pt[1],start_pt[2],r_rela,delta_L=0.0375)
    wpose.position.x = start_pt_EE[0]
    wpose.position.y = start_pt_EE[1]
    wpose.position.z = start_pt_EE[2]

    ## [bucket] return to start pt before any paths
    waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    ur_control.group.execute(plan, wait=True)
    rospy.sleep(0.5)

    if isRecord == 1:
        recorder = BagRecorder(bagPath)
        recorder.start()
        rospy.sleep(0.5)
    
    cur_path_id = 0
    ## record the start x,y (i.e., current pos) in UR frame (world frame in sand box)
    wpose = ur_control.group.get_current_pose().pose
    x_s_wldf = wpose.position.x
    y_s_wldf = wpose.position.y
    ## goal
    x_e_wldf = x_s_wldf
    y_e_wldf = y_s_wldf #+ 0.23
    
    rela_x_ls = [] # relative x 
    rela_z_ls = []       
    ite = 1 # ite starts from 1

    ## [bucket] generate waypts along bucket path
    bucketVelScale=1.0
    trajFolderName = '/scaled_trajs_060' # '/scaled_trajs_30cm'
    fileName = '/bucket_amount_goal_'+str(amount_goal)+'_pos_goal_'+str(pos_goal)+'_waterline_'+str(waterline)+'_seed_0_error_*.pkl' # unit: mm
    file_dir = DataPKLPath+trajFolderName+fileName

    execute = False
    waypts = urGivenPath4(ur_control,file_dir,cur_path_id,oigin_pt,oigin_angle_rad,0,110)
    (plan, fraction) = ur_control.go_cartesian_path2(waypts,execute=execute,velscale=bucketVelScale)
    if execute == False:
        listener.clear_finish_flag()
        listener.zero_ft_sensor()
        ur_control.set_speed_slider(bucketVelScale)
        ur_control.group.execute(plan, wait=False)

    ## --- [force monitor] ---
    rospy.loginfo('clear_finish_flag')
    flargeFlag = False
    pre_forward_dist = 0.0
    while not listener.read_finish_flag():
        rospy.sleep(0.001)               
        ## measure the force val/dir
        # f_val = listener.get_force_val()
        f_dir = listener.get_force_dir()
        _,f_val_raw_x,f_val_raw_y = listener.get_force_val_xy()
        f_val,f_val_filt_x,f_val_filt_y = listener.get_force_val_xy_filtered()

        if f_val is not None:                
            ## most conservative way (most safe)
            if np.round(f_val,6) > CUR_SAFE_FORCE:
                rospy.loginfo('[{}] ==== [MaxForce] Warning ==== \n'.format(ite-1))
                ur_control.group.stop()
                flargeFlag = True
                break 
                
            ## log list
            cur_pos = ur_control.group.get_current_pose().pose
            listener.pub_ee_pose(cur_pos)
            curx = cur_pos.position.x
            curz = cur_pos.position.z
            
            rela_x_ls.append(round(curx - oigin_pt[0],4))
            rela_z_ls.append(round(curz - oigin_pt[2],4))

            ite = ite+1            
    ## end of while loop

    now_date = time.strftime("%m%d%H%M%S", time.localtime())
    ## log (external)
    if isSaveForce ==  1:
        ## log: x_rela, z_rela, force val, force dir             
        allData = zip(rela_x_ls,rela_z_ls)                           
        with open('{}/{}_target{}_path{}_rela_xz.csv'.format(dataPath,now_date,amount_goal,cur_path_id),'a',newline="\n")as f:
            f_csv = csv.writer(f) # <<<<<<
            for row in allData:
                f_csv.writerow(row)
        f.close()
    
    # lift up
    if flargeFlag == True:
        waypoints = []
        wpose = ur_control.group.get_current_pose().pose
        wpose.position.z += 0.16
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
        ur_control.set_speed_slider(0.3)
        rospy.sleep(.5)
        ur_control.group.execute(plan, wait=True)
    
    if needScale == 1 and flargeFlag != True:
        rospy.sleep(.5)
        ur_control.set_speed_slider(0.6)
        waypoints_dump = []
        wpose = ur_control.group.get_current_pose().pose

        # keep the bucket horizonally
        wpose.orientation.x = -0.011323285790904978
        wpose.orientation.y = 0.8424439772689742
        wpose.orientation.z = -0.011348856801593265
        wpose.orientation.w = 0.5385453850989964
        waypoints_dump.append(copy.deepcopy(wpose))

        # lift up
        wpose.position.z = z_pos_scale+0.1
        waypoints_dump.append(copy.deepcopy(wpose))        

        ## move to scale overhead
        wpose.position.x = x_pos_scale
        wpose.position.y = y_pos_scale
        wpose.orientation.x = 0.6243280707459916
        wpose.orientation.y = 0.623661472464123
        wpose.orientation.z = -0.33288950702266373
        wpose.orientation.w =  0.33233327241893473
        waypoints_dump.append(copy.deepcopy(wpose))        

        ## going down to dump pt
        wpose.position.z = z_pos_scale
        waypoints_dump.append(copy.deepcopy(wpose))

        ## dump
        r_dump = R.from_euler('yz', [180+45, -90], degrees=True)
        quat_dump = r_dump.as_quat()
        wpose.orientation.x = quat_dump[0]
        wpose.orientation.y = quat_dump[1]
        wpose.orientation.z = quat_dump[2]
        wpose.orientation.w = quat_dump[3]
        waypoints_dump.append(copy.deepcopy(wpose))

        ## execute
        (plan, fraction) = ur_control.go_cartesian_path(waypoints_dump,execute=False)
        ur_control.group.execute(plan, wait=True)
        rospy.sleep(5)

        ## move up 18cm
        waypoints_dump = []
        wpose.position.z = start_pt[2]+0.18-0.1#+0.067
        waypoints_dump.append(copy.deepcopy(wpose))
        (plan, fraction) = ur_control.go_cartesian_path(waypoints_dump,execute=False)
        
        ur_control.group.execute(plan, wait=True)

    rospy.loginfo(f'Done! AG{amount_goal}_PG{pos_goal}_WL{waterline}')

    # if isRecord == 1:
    #     recorder.stop()

    # lift up
    # waypoints = []
    # wpose = ur_control.group.get_current_pose().pose
    # wpose.position.z = sp.SAFEZ
    # waypoints.append(copy.deepcopy(wpose))
    # (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    # ur_control.group.execute(plan, wait=True)    
    
    rospy.loginfo('shut down')

    if isRecord == 1:
        recorder.stop()
