# exp: ur5 + path data from the folder 'data'
# for scaled data
# 
#
# Z. Zhang
# 1/23

from cmath import cos
import copy
import threading
import time

# import apriltag_ros.msg
import numpy as np
from sqlalchemy import true
from functions.bagRecorder import BagRecorder
import rospy
from geometry_msgs.msg import PoseArray, TransformStamped
from std_msgs.msg import String

from tf import transformations as tfs
from functions.scene_helper import zero_ft_sensor,ft_listener,probe
from functions.ur_move import MoveGroupPythonInteface,goPeneGivenPoseSP,goPeneGivenPose2SP,go2OriginSP,go2GivenPoseSP
from robotiq_ft_sensor.msg import ft_sensor
from control_msgs.msg import FollowJointTrajectoryActionResult as rlst
import moveit_commander
import sys
import csv
from functions.jamming_detector import JDLib
from functions.gpr_helper import GPRPredict
from functions.handle_drag_force import smooth_fd_kf, get_mean
from functions.drawTraj import urCentOLine,urCentOLine_sim,urCent2Circle,urCircle,urCircleLine
from functions.saftyCheck import saftyCheckHardSP
from functions.saftyCheck import SfatyPara15LBoxSand
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import RBF,Matern,ExpSineSquared,WhiteKernel
from functions.boa_helper import plot_2d2,plot_2d3,plot_2d4
from sklearn.linear_model import LinearRegression

import pickle
from scipy.spatial.transform import Rotation as R

## only x,y,z positions along paths
def urGivenPath(ur_control,file_dir,path_id,oigin_pt):
    ## ur
    waypoints1 = []
    wpose = ur_control.group.get_current_pose().pose

    ## given path info
    f = open(file_dir,'rb')
    data = pickle.load(f)
    paths_xyz = data['loader_pos_trajs']
    paths_theta = data['loader_rot_trajs']
    num_path = np.shape(paths_xyz)[0]
    path_length = data['episode_length']

    ## path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    ## theta (rad) R^{75*1}
    rot_traj = paths_theta[path_id,:,:]

    for i in range(37):
        x_shiyu = cur_path_xyz[i,0]
        y_shiyu = cur_path_xyz[i,1]
        z_shiyu = cur_path_xyz[i,2]
        theta_deg_shiyu = np.degrees(rot_traj[i,0])

        ## convert shiyu frame to UR frame (global)
        x_global = z_shiyu + oigin_pt[0]
        y_global = -x_shiyu+ oigin_pt[1]
        z_global = y_shiyu + oigin_pt[2]

        wpose.position.x = x_global
        wpose.position.y = y_global
        wpose.position.z = z_global        
        waypoints1.append(copy.deepcopy(wpose))
    
    return waypoints1

## with varying angles along paths
def urGivenPath2(ur_control,file_dir,path_id,oigin_pt,oigin_angle_rad):
    ## ur
    waypoints1 = []
    wpose = ur_control.group.get_current_pose().pose

    ## given path info
    f = open(file_dir,'rb')
    data = pickle.load(f)
    paths_xyz = data['loader_pos_trajs']
    paths_theta = data['loader_rot_trajs']
    num_path = np.shape(paths_xyz)[0]
    path_length = data['episode_length']

    ## path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    ## theta (rad) R^{75*1}
    rot_traj = paths_theta[path_id,:,:]

    for i in range(37):
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

        wpose.position.x = x_global
        wpose.position.y = y_global
        wpose.position.z = z_global        

        ## quaternion in global frame
        theta_rad_global = -theta_rad_shiyu
        r = R.from_euler('y', theta_rad_global+oigin_angle_rad)
        # r_rotmat = R.from_euler('y', theta_rad_global).as_matrix()
        quat_global = r.as_quat()
        
        wpose.orientation.x = quat_global[0]
        wpose.orientation.y = quat_global[1]
        wpose.orientation.z = quat_global[2]
        wpose.orientation.w = quat_global[3]

        waypoints1.append(copy.deepcopy(wpose))
    return waypoints1

## with varying angles along paths
## with given iteration interval [ite_start,ite_end)
def urGivenPath3(ur_control,file_dir,path_id,oigin_pt,oigin_angle_rad,ite_start,ite_end):
    ## ur
    waypoints1 = []
    wpose = ur_control.group.get_current_pose().pose

    ## given path info
    f = open(file_dir,'rb')
    data = pickle.load(f)
    paths_xyz = data['loader_pos_trajs']
    paths_theta = data['loader_rot_trajs']
    num_path = np.shape(paths_xyz)[0]
    path_length = data['episode_length']

    if path_id >= num_path:
        raise Exception('path id error')

    ## path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    ## theta (rad) R^{75*1}
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

        wpose.position.x = x_global
        wpose.position.y = y_global
        wpose.position.z = z_global        

        ## quaternion in global frame
        theta_rad_global = -theta_rad_shiyu
        r = R.from_euler('y', theta_rad_global+oigin_angle_rad)
        # r_rotmat = R.from_euler('y', theta_rad_global).as_matrix()
        quat_global = r.as_quat()
        
        wpose.orientation.x = quat_global[0]
        wpose.orientation.y = quat_global[1]
        wpose.orientation.z = quat_global[2]
        wpose.orientation.w = quat_global[3]

        waypoints1.append(copy.deepcopy(wpose))
    return waypoints1

# to read 'scaled_xxx' files
# unit: change to m
# considering delta offset from FT300 sensor (0.0375m)
def urGivenPath4(ur_control,file_dir,path_id,oigin_pt,oigin_angle_rad,ite_start,ite_end):
    ## ur
    waypoints1 = []
    wpose = ur_control.group.get_current_pose().pose

    ## given path info
    f = open(file_dir,'rb')
    data = pickle.load(f)
    paths_xyz = data['loader_pos_trajs']/1000 # unit: change to m
    paths_theta = data['loader_rot_trajs']
    num_path = np.shape(paths_xyz)[0]
    path_length = data['episode_length']

    if path_id >= num_path:
        raise Exception('path id error')

    ## path xyz R^{75*3}
    cur_path_xyz = paths_xyz[path_id,:,:]
    ## theta (rad) R^{75*1}
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
    input = int(input("Exp Mode: trial[0], normal(1) ") or "0")
    if input == 0:
        exp_mode = 'trial'
    elif input == 1:
        exp_mode = 'normal'
    else:
        raise Exception('Error: Invalid Exp Mode!')
    print('**** Exp Mode: '+exp_mode+' ****')
    rospy.sleep(0.5)
    ## set the initial pos (i.e., origin of task frame)
    sp = SfatyPara15LBoxSand()
    originx = sp.originX
    originy = sp.originY
    originz = sp.originZ

    ##--- BOA related codes ---#
    # kernel = RBF(length_scale=8, length_scale_bounds='fixed')
    # kernel = Matern(length_scale=1, length_scale_bounds='fixed',nu=np.inf)
    lenScaleBound ='fixed'
    # lenScaleBound = (1e-5, 1e5)
    # lenScaleBound = (0.01, 0.2)
    kernel = Matern(length_scale=0.1, length_scale_bounds=lenScaleBound, nu=np.inf)
    # kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=np.inf)
    # kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=2.5)
    # kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=1.5)
    str_kernel = str(kernel)

    ## initial distribution in BOA
    xrange = np.linspace(sp.xmin, sp.xmax, sp.xBoaSteps)
    yrange = np.linspace(sp.ymin, sp.ymax, sp.yBoaSteps)
    X, Y = np.meshgrid(xrange, yrange)
    xrange = X.ravel()
    yrange = Y.ravel()
    XY = np.vstack([xrange, yrange]).T
    ##=== BOA related codes ===#

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

    ## set zero to the ft 300
    zero_ft_sensor()
    rospy.sleep(0.5) 

    Lrang = 0.3 # <<<<<<
    ## position of buried objects
    ds_obj = 0.24
    
    LIFT_HEIGHT = +0.10 #(default: +0.10) # <<<<<<
    # saftz = initPtz + LIFT_HEIGHT
    # PENETRATION DEPTH
    if exp_mode == 'trial':
        PENE_DEPTH = 0.05    #(default: -0.03) # <<<<<<
        normalVelScale = 0.1 # <<<<<<
    elif exp_mode == 'normal':
        PENE_DEPTH = -0.04   #(default: -0.03) # <<<<<<
        normalVelScale = 0.1 #(default: 0.2) <<<<<<
    else:
        raise Exception('Error: Invalid Exp Mode!')    
    depthz = originz + PENE_DEPTH
    maxVelScale    = 0.3 # <<<<<<
    # Cur SAFE FORCE
    CUR_SAFE_FORCE = 15.0  #(default: 15N) # <<<<<<
    
    # folder name
    expFolderName = '/20230115_exp' # <<<<<<
    NutStorePath = '/home/zhangzeqing/Nutstore Files/Nutstore/water_manipulate'
    dataPath = NutStorePath+expFolderName+'/data'
    figPath = NutStorePath+expFolderName+'/fig'
    bagPath = '/home/zhangzeqing/ur5_ws/src/ur_control/logs' #'/home/zhangzeqing/rosbag'
    isSaveForce = 1           # <<<<<<
    isRecord = 1 # rosbag recorder
    isPlotJD = 1
    ## JD setting
    ite_bar = 30
    delta_ite = 10
    ds_min = 0.005
    JDid = 1
    diff_bar = 0.5 # N # <<<<<<
    ## buried object shape
    obj1 = np.array([[0.035,0.175],
    [0.035,0.255],
    [0.115,0.175]])
    obj2 = np.array([[0.155,0.085],
    [0.207,0.141],
    [0.237,0.116],
    [0.185,0.06]])
    object_shape = [obj1, obj2]

    listener = ft_listener()
    probe = probe()
    ## BOA init. (bounds in the relatvie frame)
    bo = BayesianOptimization(f=None, pbounds={'x': (sp.xmin, sp.xmax), 'y': (sp.ymin, sp.ymax)},
                        verbose=2,
                        random_state=1)
    # plt.ioff()
    bo.set_gp_params(kernel=kernel)
    util = UtilityFunction(kind="ei", 
                        kappa = 2, 
                        xi=0.5,
                        kappa_decay=1,
                        kappa_decay_delay=0)
    
    ## velocity setting
    ur_control.set_speed_slider(maxVelScale)

    ## check exp safety setting at the beginning
    if saftyCheckHardSP(sp,LIFT_HEIGHT,PENE_DEPTH,CUR_SAFE_FORCE):
        print('***** Safety Check Successfully *****')
    else:
        raise Exception('Error: Safety Check Failed')
    
    ## Calibration: go the origin
    # go2OriginSP(ur_control,sp)
    
    ## go to test points
    # test_pose = [0 for hh in range(0,3)]
    # test_pose[0] = originx +0.085
    # test_pose[1] = originy
    # test_pose[2] = originz
    # go2GivenPoseSP(ur_control,sp,test_pose)
    # rospy.sleep(0.5)

    ## go the init pos of the exp 
    # init_pose = [0 for hh in range(0,3)]
    # init_pose[0] = originx + 0.085 - 0.005*6
    # init_pose[1] = originy + 0.1
    # init_pose[2] = sp.SAFEZ
    # go2GivenPoseSP(ur_control,sp,init_pose)
    # rospy.sleep(0.5)

    ## [bucket] start point x=0,y=0.45,z=0 expressed in shiyu frame 
    start_pt = [-0.54943859455702817, 0.10205824512513274,0.08795793366304825]
    start_pt[1] -= 0.05
    start_pt[2] += 0.05
    waypoints = []
    wpose = ur_control.group.get_current_pose().pose
    # wpose.position.x = start_pt[0]#+0.21 #-0.1
    # wpose.position.y = start_pt[1]
    # wpose.position.z = start_pt[2]#+0.2 #+0.2 #-0.25
    ## [bucket] oigin of shiyu frame expressed in global frame
    oigin_pt = np.array(start_pt)-np.array([0,0,0.3])

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
    wpose.position.x = start_pt_EE[0]#+0.21 #-0.1
    wpose.position.y = start_pt_EE[1]
    wpose.position.z = start_pt_EE[2]#+0.2 #+0.2 #-0.25

    ## [bucket] return to start pt before any paths
    waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    ur_control.group.execute(plan, wait=True)
    rospy.sleep(0.5)
    
    
    fd_nonjamming = 3  # 3N
    fd_object = 7  # 3N
    traj_radius = 0.01 # xx cm

    # switch: w/ or w/o proximity sensing
    wSensing = 1

    ## parameters of GPR
    # seasonal_kernel = ExpSineSquared(length_scale=1, periodicity=1200, periodicity_bounds=(1000, 1500))
    seasonal_kernel = ExpSineSquared(length_scale=1, periodicity=600, periodicity_bounds=(400, 700))
    noise_kernel = WhiteKernel(noise_level=1e-1)
    gprKernel = seasonal_kernel + noise_kernel
    delta_ite_train = 2000 #<<<<< optimal
    delta_ite_test = 1000
    everyPts = 20 # downsample
    gprp = GPRPredict(gprKernel,delta_ite_train,delta_ite_test,everyPts)
    pre_window_index = int(0)

    if isRecord == 1:
        recorder = BagRecorder(bagPath)
        recorder.start()
        rospy.sleep(0.5)

    ## start the loop
    for cur_path_id in range(0,1): # <<<<<<
        print("--------- {}-th path ---------".format(cur_path_id))
        ## record the start x,y (i.e., current pos) in UR frame (world frame in sand box)
        wpose = ur_control.group.get_current_pose().pose
        x_s_wldf = wpose.position.x
        y_s_wldf = wpose.position.y

        ######### given goal ##########
        x_e_wldf = x_s_wldf
        y_e_wldf = y_s_wldf #+ 0.23

        ## list to record the df, dr, ds
        df_ls = []
        dr_ls = []
        df_raw_x_ls = [] # w/o round
        df_raw_y_ls = [] # w/o round
        df_filt_x_ls = [] # filtered
        df_filt_y_ls = [] # filtered
        ds_ls = []    
        ds_ite_ls = []
        maxForward_ls = []  
        rela_x_ls = [] # relative x 
        rela_z_ls = []
        boa_ite_ls = []
        boa_x_ls = []
        boa_y_ls = []
        boa_return_ls = []
        ite_out95_ls = [] # total # outliers in the whole process
        ite_out99_ls = [] # total # outliers in the whole process
        outlier_log_ls = []
        time_cost_pgrp = []
        ## define outlier counter
        curEps_cntOut95 = 0
        curEps_cntOut99 = 0
        ## define outlier/z-score/LRslope list for each window
        curEps_ite_out99_ls = []     
        curEps_zscore_out99_ls = [] 
        curEps_LRslope_ls = []
        ## list: 2d velocity of EE
        vel2d_ls = []

        ## initialize parameters for each slide        
        ite = 1 # ite starts from 1, exp iteration starts from 0
        cent_dist = 0
        cntOut95Bar = 10 # safety threshold
        cntOut99Bar = 7  # safety threshold
        bar_curEps_cntOut99 = 3 # bar num to do LR
        bar_curBat_cntOut99 = 3 # bar num to do LR in one batch
        batch_size = 100 # ite size of one batch, do RL for outliers with each batch
        ignore_upto = 260 # ignore [0,ignore_upto] data due to large force at beginning of probe motion
        ignore_upto = int(np.floor(ignore_upto/gprp.everyPts)*gprp.everyPts)
        ## jamming detection parameters
        jc_zscore_bar = 4.5 #<<<< NEW ADD
        jdcond2 = 0 #<<<< NEW ADD

        ## [bucket] generate waypts along bucket path
        bucketVelScale=0.3
        expFolderName = '/home/zhangzeqing/ur5_ws/src/ur_water_bucket/data'
        fileName = '/scaled_bucket_targeted_amount_0.6_saved_trajs.pkl'
        file_dir = expFolderName+fileName
        path_id = 0

        # # waypts = urGivenPath(ur_control,file_dir,cur_path_id,oigin_pt)
        # waypts = urGivenPath2(ur_control,file_dir,cur_path_id,oigin_pt,oigin_angle_rad)
        # ## [bucket] plan & execute
        # (plan, fraction) = ur_control.go_cartesian_path(waypts,execute=False)
        # listener.clear_finish_flag()
        # ur_control.set_speed_slider(bucketVelScale)
        # listener.zero_ft_sensor()
        # rospy.sleep(0.5)
        # ur_control.group.execute(plan, wait=False)
        
        
        execute = False
        waypts = urGivenPath4(ur_control,file_dir,cur_path_id,oigin_pt,oigin_angle_rad,0,37)
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
            with open('{}/{}_path{}_rela_xz.csv'.format(dataPath,now_date,cur_path_id),'a',newline="\n")as f:
                f_csv = csv.writer(f) # <<<<<<
                for row in allData:
                    f_csv.writerow(row)
            f.close()
        
        # lift up
        if flargeFlag == True:
            waypoints = []
            wpose = ur_control.group.get_current_pose().pose
            wpose.position.z = sp.SAFEZ
            waypoints.append(copy.deepcopy(wpose))
            (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
            ur_control.group.execute(plan, wait=True)
        
        rospy.loginfo('{}-th path finished'.format(cur_path_id))
    # end of for-loop

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
