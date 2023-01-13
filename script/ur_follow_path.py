# exp: ur5 + path data from the folder 'data'
#
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

    for i in range(9):       
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
    expFolderName = '/20230112_UR_EEVel_circled' # <<<<<<
    NutStorePath = '/home/zhangzeqing/Nutstore Files/Nutstore/02Wedge'
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
    waypoints = []
    wpose = ur_control.group.get_current_pose().pose
    wpose.position.x = start_pt[0]#+0.21 #-0.1
    wpose.position.y = start_pt[1]
    wpose.position.z = start_pt[2]#+0.2 #+0.2 #-0.25
    waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    ur_control.group.execute(plan, wait=True)
    rospy.sleep(0.5)
    ## [bucket] oigin of shiyu frame expressed in global frame
    oigin_pt = np.array(start_pt)-np.array([0,0,0.45])

    
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
    for slide_id in range(1,2): # <<<<<<
        print("--------- {}-th slide ---------".format(slide_id))
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
        rela_y_ls = []
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
        ## goal
        # x_e_wldf = initPtx + 0.12
        # y_e_wldf = initPty + 0.3

        ## circle+line (a.k.a. spiral traj.)
        # region
        # # _,_,waypts = urCentOLine(ur_control,0.01,0.01,[x_e_wldf,y_e_wldf])
        # # _,_,waypts = urCentOLine_sim(ur_control,traj_radius,0.01,[x_e_wldf,y_e_wldf])
        # # waypts = urCentOLine2(ur_control,traj_radius,0.01,[x_s_wldf,y_s_wldf],[x_e_wldf,y_e_wldf])
        # waypts = urCircleLine(ur_control,traj_radius,0.01,[x_s_wldf,y_s_wldf],[x_e_wldf,y_e_wldf])
        # # _,_,Ocent,waypts = urCent2Circle(ur_control,traj_radius,1,False)
        # (plan, fraction) = ur_control.go_cartesian_path(waypts,execute=False)
        # ## move along the generated path
        # listener.clear_finish_flag()
        # ur_control.set_speed_slider(normalVelScale)
        # listener.zero_ft_sensor()
        # rospy.sleep(0.5)
        # ur_control.group.execute(plan, wait=False)
        # endregion

        # go to the goal (line)
        # region
        # linearVelScale = 0.1
        # ur_control.set_speed_slider(linearVelScale)
        # waypoints = []
        # wpose.position.x = x_e_wldf
        # wpose.position.y = y_e_wldf
        # waypoints.append(copy.deepcopy(wpose))
        # (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
        # listener.clear_finish_flag()
        # listener.zero_ft_sensor()
        # rospy.sleep(0.5)
        # ur_control.group.execute(plan, wait=False)
        # endregion

        ## circular motion 
        # region
        # circleVelScale=0.3
        # num_circle = 5
        # waypts = urCircle(ur_control,traj_radius,[x_s_wldf,y_s_wldf],num_circle)
        # (plan, fraction) = ur_control.go_cartesian_path(waypts,execute=False)
        # listener.clear_finish_flag()
        # ur_control.set_speed_slider(circleVelScale)
        # listener.zero_ft_sensor()
        # rospy.sleep(0.5)
        # ur_control.group.execute(plan, wait=False)
        # endregion

        ## bucket path
        bucketVelScale=0.1
        expFolderName = '/home/zhangzeqing/ur5_ws/src/ur_water_bucket/data'
        fileName = '/bucket_targeted_amount_0.6_saved_trajs.pkl'
        file_dir = expFolderName+fileName
        path_id = 0
        waypts = urGivenPath(ur_control,file_dir,path_id,oigin_pt)
        (plan, fraction) = ur_control.go_cartesian_path(waypts,execute=False)
        listener.clear_finish_flag()
        ur_control.set_speed_slider(bucketVelScale)
        listener.zero_ft_sensor()
        rospy.sleep(0.5)
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
            ## measure the tip velocities (6D)
            # eetwist = ur_control.get_ee_twist()
            # print('twist:{}'.format(eetwist))
            # vx = eetwist[0]
            # vy = eetwist[1]
            # vxy = np.sqrt(vx**2+vy**2)
            # print(f'vel xy: {vxy}')
            # vel2d = np.linalg.norm(eetwist[:2])
            # print(f'vel 2d: {vel2d}')

            if f_val is not None:                
                ## most conservative way (most safe)
                if np.round(f_val,6) > CUR_SAFE_FORCE:
                    rospy.loginfo('[{}] ==== [MaxForce] Warning ==== \n'.format(ite-1))
                    ur_control.group.stop()
                    flargeFlag = True
                    ## pub probe state (get force threshold)
                    probe.pub_probe_state(probe.state.FORCE_THRESHOLD)
                    break 
                ite = ite+1
            
        ## end of while loop
        ## record outlier info of last episode
        curEps_outlier_log_ls = [curEps_cntOut95, curEps_cntOut99, \
                            len(curEps_ite_out99_ls),curEps_ite_out99_ls, \
                            len(curEps_zscore_out99_ls),curEps_zscore_out99_ls,\
                            len(curEps_LRslope_ls),curEps_LRslope_ls]
        outlier_log_ls.append(curEps_outlier_log_ls)
        curEps_outlier_log_ls = []

        # record last situation when meet the large force
        if flargeFlag == 1:
            df_ls.append(round(f_val,4))
            dr_ls.append(round(f_dir,4))
            df_raw_x_ls.append(f_val_raw_x)
            df_raw_y_ls.append(f_val_raw_y)
            df_filt_x_ls.append(f_val_filt_x)
            df_filt_y_ls.append(f_val_filt_y)
            cur_pos = ur_control.group.get_current_pose().pose
            curx = cur_pos.position.x
            cury = cur_pos.position.y
            rela_x_ls.append(round(curx - originx,4))
            rela_y_ls.append(round(cury - originy,4))

        # go back to the goal (if use urCircleLine() and complete the whole spiral traj.)
        # if flargeFlag == 0:
        #     waypoints_goal = []
        #     wpose = ur_control.group.get_current_pose().pose
        #     wpose.position.x = x_e_wldf
        #     wpose.position.y = y_e_wldf
        #     waypoints_goal.append(copy.deepcopy(wpose))
        #     (plan, fraction) = ur_control.go_cartesian_path(waypoints_goal,execute=False)
        #     ur_control.group.execute(plan, wait=True)

        now_date = time.strftime("%m%d%H%M%S", time.localtime())
        ## log (external)
        if isSaveForce ==  1:
            ## log: x_rela, y_rela, force val, force dir             
            allData = zip(rela_x_ls,rela_y_ls,df_ls,dr_ls,vel2d_ls)                           
            with open('{}/{}_slide{}_Fdvaldirvel.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
                f_csv = csv.writer(f) # <<<<<<
                for row in allData:
                    f_csv.writerow(row)
            f.close()

            ## log: ite - center distance
            allData = zip(ds_ite_ls,ds_ls)
            with open('{}/{}_slide{}_Distance.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
                f_csv = csv.writer(f) # <<<<<<
                for row in allData:
                    f_csv.writerow(row)
            f.close()

            ## log: 4 info. on BOA                
            allData = zip(boa_ite_ls,boa_x_ls,boa_y_ls,boa_return_ls)
            with open('{}/{}_slide{}_BOA.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
                f_csv = csv.writer(f) # <<<<<<
                ## record the start and goal (relative)
                tempRow = [x_s_wldf-originx, y_s_wldf-originy, x_e_wldf-originx, y_e_wldf-originy]
                f_csv.writerow(tempRow)
                for row in allData:
                    f_csv.writerow(row)
            f.close()

            ## log: x_rela, y_rela, Fx_raw, Fy_raw, ForceDir
            # allData = zip(rela_x_ls,rela_y_ls,df_raw_x_ls,df_raw_y_ls,dr_ls)     
            # with open('{}/{}_slide{}_xyFxFydr.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
            #     f_csv = csv.writer(f) # <<<<<<
            #     for row in allData:
            #         f_csv.writerow(row)
            # f.close()

            ## log: iteration records for outliers occur    
            # with open('{}/{}_slide{}_iteOutliers.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
            #     f_csv = csv.writer(f) # <<<<<<
            #     f_csv.writerow(ite_out95_ls)
            #     f_csv.writerow(ite_out99_ls)
            # f.close()

            with open('{}/{}_slide{}_Outliers.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
                f_csv = csv.writer(f) # <<<<<<
                ## total # outliers during the whole process
                f_csv.writerow(ite_out95_ls)
                f_csv.writerow(ite_out99_ls)                  
            f.close()            

            ## log: Fx,Fy_raw,_filtered
            # allData = zip(df_raw_x_ls,df_raw_y_ls,df_filt_x_ls,df_filt_y_ls)     
            # with open('{}/{}_slide{}_FxFyrawfilt.csv'.format(dataPath,now_date,slide_id),'a',newline="\n")as f:
            #     f_csv = csv.writer(f) # <<<<<<
            #     for row in allData:
            #         f_csv.writerow(row)
            # f.close()

        ## if no jamming, plot it， and ds_ls not empty
        # if isPlotJD and not flargeFlag and ds_ls:
        #         ds_adv = round(ds_obj-ds_ls[-1], 3) # >0 in theory
        #         title_str = 'Exp{}: ds [{},{}], Dep {}, Vel {}, Ite {}, NoJD'.format(slide_id,ds_min,np.inf,PENE_DEPTH,normalVelScale,len(df_ls))
        #         JDlib.plotJDRes(ds_obj,title_str,figPath,slide_id)

        ## plot BOA results
        # tempCurPos = ur_control.group.get_current_pose().pose.position
        # probeCurPt = [tempCurPos.x-originx,tempCurPos.y-originy]
        ## Def: plot_2d2(slide_id, bo, util, kernel,x,y,XY, f_max, fig_path, name=None)
        # plot_2d4(slide_id, bo, util, kernel, xrange,yrange,XY, CUR_SAFE_FORCE, figPath+'/{}_slide{}_'.format(now_date,slide_id),
        #         probeCurPt, object_shape, "{:03}".format(len(bo._space.params)))
        
        rospy.loginfo('{}-th slide finished'.format(slide_id))

    # if isRecord == 1:
    #     recorder.stop()

    # lift up
    waypoints = []
    wpose = ur_control.group.get_current_pose().pose
    wpose.position.z = sp.SAFEZ
    waypoints.append(copy.deepcopy(wpose))
    (plan, fraction) = ur_control.go_cartesian_path(waypoints,execute=False)
    ur_control.group.execute(plan, wait=True)    
    
    rospy.loginfo('shut down')

    if isRecord == 1:
        recorder.stop()
