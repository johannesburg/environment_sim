#!/usr/bin/env python
# ROS imports
import rospy
import range_libc
from geometry_msgs.msg import PoseStamped, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap

import utils as Utils
# Compute tools
import numpy as np
try:
  import torch
  import torch.utils.data
  from torch.autograd import Variable
  import torch.nn.functional as F
except ImportError:
  print('No torch module found')

class EnvironmentSim:
    
    def __init__(self):
        self.GT_INDEX = 0               # Ground truth index (for sampling in K > 1 poses)
        # Scan constants
        self.THETA_DISCRETIZATION = 112 # Discretization of scanning angle
        self.Z_SHORT = 0.01 #0.14       # Weight for short reading
        self.LAMBDA_SHORT = 0.05        # Intrinsic parameter of the exponential dist.
        self.Z_MAX = 0.07 #0.015        # Weight for max reading
        self.Z_RAND = 0.12 #0.045       # Weight for random reading
        self.SIGMA_HIT = 8.0            # Noise value for hit reading
        self.Z_HIT = 0.80 #0.80         # Weight for hit reading
        self.MAX_RANGE_METERS = float(rospy.get_param("~max_range_meters", 5.6)) # The max range of the laser
        
        # Dynamics constants
        self.MIN_VEL = -0.75 #TODO make sure these are right
        self.MAX_VEL = 0.75
        self.MIN_DEL = -0.34
        self.MAX_DEL = 0.34

        self.laser_params = { "angle_min": -2.08621382713,
                              "angle_max": 2.09234976768,
                        "angle_increment": 0.00613592332229,
                         "time_increment": 9.76562732831e-05,
                              "scan_time": 0.10000000149,
                              "range_min": 0.019999999553,
                              "range_max": 5.59999990463}
        self.num_ranges = 682                    # Number of ranges to publish (M)
                                                 # This is equivalent to int(max-min)/incr + 2

        self.dtype = torch.cuda.FloatTensor

        self.K = 5                               # Number of particles to publish

                                                 # TODO: These values will need to be tuned
        self.sigma = torch.Tensor([0.25, 0.1]).type(self.dtype).expand(self.K, 2) 

                                                 # Initial position [x, y, theta]
        INITIAL_POSE = torch.Tensor([0.0, 0.0, 0.0]).type(self.dtype)           

        dt = 0.1                                 # Constant dt, same as trained model

        # TODO: get topic names from server
        self.pose_pub_topic = "/sim/ground_truth/pose" 
        self.poses_pub_topic = "/sim/ground_truth/poses"  # TODO get rid of poses / or create a ROI
        self.scan_pub_topic = "/sim/scan"

        self.set_pose_sub_topic = "/sim/set_pose"
        self.control_sub_topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"

        #self.model_name = rospy.get_param("~nn_model", "/media/JetsonSSD/model3.torch")
        self.model_name = rospkg.RosPack().get_path("environment_sim") + "/models/dynamics_model_noisy.torch"

        # Initialize publishers and subscribers
        self.pose_pub = rospy.Publisher(self.pose_pub_topic, PoseStamped, queue_size=10)
        self.poses_pub = rospy.Publisher(self.poses_pub_topic, PoseArray, queue_size=10)
        self.scan_pub = rospy.Publisher(self.scan_pub_topic, LaserScan, queue_size=1)

        self.model = torch.load(self.model_name)
        self.model.cuda()                        # Tell torch to run the network on the GPU
        self.model.eval()                        # Model ideally runs faster in eval mode

        print("Loading:", self.model_name)
        print("Model:\n",self.model)
        print("Torch Datatype:", self.dtype)

        self.noise     = torch.Tensor(self.K, 2).type(self.dtype) # (K,2)
        self.ctrl      = torch.zeros(self.K,2).type(self.dtype)   # (K,2)
        self.pose      = INITIAL_POSE.repeat(self.K, 1)           # (K,3)
        self.pose_dot  = torch.zeros(self.K, 3).type(self.dtype)  # (K,3)
        self.nn_input  = torch.zeros(self.K, 8).type(self.dtype)  # (K,8)
        self.nn_input[:,4] = 1.0 # cos(0)
        self.nn_input[:,7] = dt  # set dt = 0.1

        # 0. Wait for map to publish / static map server
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('/static_map', GetMap)
        map_msg = static_map().map

        self.ranges = np.zeros(self.num_ranges, dtype=np.float32)
        self.query = np.zeros((1,3), dtype=np.float32)            # (num particles = 1, 3)
        self.obs_angles = np.arange(self.laser_params["angle_min"],
                self.laser_params["angle_max"] + self.laser_params["angle_increment"], 
                self.laser_params["angle_increment"], 
                dtype=np.float32)

        oMap = range_libc.PyOMap(map_msg) # A version of the map that range_libc can understand
        max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution) # The max range in pixels of the laser
        self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, self.THETA_DISCRETIZATION) # The range method that will be used for ray casting
        # TODO: do we need this to create a range
        self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px)) # Load the sensor model expressed as a table 

        self.control_sub = rospy.Subscriber(self.control_sub_topic, 
                AckermannDriveStamped, self.control_cb, queue_size=10)

        self.set_pose_sub = rospy.Subscriber(self.set_pose_sub_topic, 
                PoseStamped, self.set_pose_cb, queue_size=1)

        # Testing Code
        # 1. Publish initial GT pose(s) to EnvSim
            # TODO: publish initial pose
        # 2. Publish initial pose estimate to PF
            # TODO: publish initial pose
        # 3. Publish goal to MPPI
            # TODO: publish goal
        # 4. Log experiment data
        # 5. On TIMEOUT or GOAL IS REACHED
            # Dump experiment logs, run next test (Goto 1.)

    def normal(self, mean, sd, x):
      return np.exp(-(1.0/2)*((x-mean)**2)/(1.0*sd**2))/(np.sqrt(2*np.pi*(sd**2)))

    def precompute_sensor_model(self, max_range_px):
       rospy.logerr("started precompute sensor model")
  
       table_width = int(max_range_px) + 1
       sensor_model_table = np.zeros((table_width,table_width))
       # Populate sensor model table as specified
       for ztk_star in range(table_width):
         hit_norm = 0 
         for ztk in range(table_width): # P_hit
           sensor_model_table[ztk,ztk_star] = self.normal(ztk_star, self.SIGMA_HIT, ztk)  
           hit_norm += sensor_model_table[ztk,ztk_star]
         sensor_model_table[:,ztk_star] *= self.Z_HIT / hit_norm
         if ztk_star != 0:
           short_norm = 1 / (1 - np.exp(-self.LAMBDA_SHORT * ztk_star))
         else:
               short_norm = 0
         for ztk in range(table_width):
           if ztk <= ztk_star:
             sensor_model_table[ztk,ztk_star] += short_norm * self.LAMBDA_SHORT * np.exp(-self.LAMBDA_SHORT * ztk) * self.Z_SHORT # P_short
           sensor_model_table[ztk,ztk_star] += self.Z_RAND / max_range_px #P_rand
           if ztk == max_range_px: 
             sensor_model_table[ztk,ztk_star] += self.Z_MAX # P_max
         sensor_model_table[:,ztk_star] /= np.sum(sensor_model_table[:,ztk_star])
  
       rospy.logerr("done precomputing sensor model")
       # plot precomputed distribution
       #spot_checks = [0, 20, 150, 200, 280]
       #for col in spot_checks:
       #  plt.scatter(np.arange(0,table_width), sensor_model_table[:,col])
       #  plt.show()
  
       return sensor_model_table

    # Receives an AckermannDriveStamped message with control output from MPPI
    def control_cb(self, msg):
        # Compute next pose(s) from control message
        # Cache results for next computation
        # Publish next pose(S)
        vel = msg.drive.speed
        delta = msg.drive.steering_angle
        # self.ctrl = msg.ctrl
        self.ctrl = torch.Tensor([vel, delta]).expand(self.K,2)
        # self.compute_next_pose()
        self.compute_next_pose()
        # Compute laserscan/ranges at new current pose
        self.compute_ranges()
        #### Prepare messages to publish

        # get pose message
        control_msg = self.poses_to_posearray()
        ps = PoseStamped()
        ps.header = control_msg.header
        ps.pose = control_msg.poses[self.GT_INDEX]
        print("vel: " + str(vel), "delta: " + str(delta))
        print(ps)

        # get scan message
        scan_msg = self.ranges_to_laserscan()

        self.pose_pub.publish(ps)
        self.poses_pub.publish(control_msg)
        self.scan_pub.publish(scan_msg)

    def set_pose_cb(self, msg):
        pose = Utils.posestamped_to_config(msg)
        print("Setting pose to: " + str(pose))
        pose = torch.Tensor(pose).type(self.dtype)           
        self.pose = pose.repeat(self.K, 1)                # (K,3)

    def compute_next_pose(self):
        # init_pose (3,) [x, y, theta]
        # init_input (8,):
        #   0    1       2          3           4        5      6   7
        # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt

        torch.normal(0, self.sigma, out=self.noise)
        # Create nn_input
        self.nn_input[:,0:3] = self.pose_dot
        self.nn_input[:,5:7] = self.ctrl                  # TODO: Add noise?
        self.nn_input[:,5:7].add_(self.noise)             # (K,2) + (K,2) # Add noise
        self.nn_input[:,5].clamp_(self.MIN_VEL, self.MAX_VEL)
        self.nn_input[:,6].clamp_(self.MIN_DEL, self.MAX_DEL)
        torch.sin(self.pose[:,2], out=self.nn_input[:,3]) # sin(theta)
        torch.cos(self.pose[:,2], out=self.nn_input[:,4]) # cos(theta)
        # pose_dot = model(nn_input)
        # Call model to learn new pose_dot
        pose_dot = self.model(Variable(self.nn_input, requires_grad=False)) # (K, 3)
        self.pose_dot = pose_dot.data
        # pose += pose_dot
        # update pose
        self.pose.add_(self.pose_dot) # Update pose
        #Utils.clamp_angle_tensor_(pose_dot[:,2])
        Utils.clamp_angle_tensor_(self.pose[:,2])

    def compute_ranges(self):
        self.query[:,:] = self.pose[self.GT_INDEX].cpu().numpy()
        self.range_method.calc_range_repeat_angles(self.query, self.obs_angles, self.ranges)

    def poses_to_posearray(self):
        msg = PoseArray()
        msg.header = Utils.make_header('map')
        msg.poses = Utils.particles_to_poses(self.pose)
        return msg

    def ranges_to_laserscan(self):
        # TODO: optimize if bottleneck
        msg = LaserScan()
        msg.header = Utils.make_header('laser')
        msg.angle_min = self.laser_params["angle_min"]
        msg.angle_max = self.laser_params["angle_max"]
        msg.angle_increment = self.laser_params["angle_increment"]
        msg.time_increment = self.laser_params["time_increment"]
        msg.scan_time = self.laser_params["scan_time"]
        msg.range_min = self.laser_params["range_min"]
        msg.range_max = self.laser_params["range_max"]
        msg.ranges = self.ranges.tolist()
        msg.intensities = self.ranges.tolist()
        return msg

if __name__ == "__main__":
  print("Initializing Environment Sim...")
  rospy.init_node("environment_sim", anonymous=True) # Initialize the node
  print("Node created!")
  pf_sim = EnvironmentSim()
  print("ESim initialized, spinning...")
  # TODO: run the laser in a tight loop
  rospy.spin()
