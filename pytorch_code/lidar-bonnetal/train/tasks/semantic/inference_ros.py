'''
Description: Do not Edit
Author: muzi2045
Date: 2021-02-23 17:22:32
LastEditor: muzi2045
LastEditTime: 2021-03-04 16:09:33
'''

import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 
import yaml

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

import torch.backends.cudnn as cudnn
from torch.nn import functional as f

from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.dataset.kitti.parser import Parser
from common.laserscan import LaserScan

def load_config(path : str):
  # open arch config file
  try:
    print("Opening arch config file from %s" % model_path)
    ARCH = yaml.safe_load(open(model_path + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % model_path)
    DATA = yaml.safe_load(open(model_path + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()
  return ARCH, DATA

class Processor_ROS:
    def __init__(self, model_path):
      self.points = None
      self.model_path = model_path
      self.device = None
      self.net = None
      self.inputs = None
        
    def initialize(self):
      self.read_config()
        
    def read_config(self):
      self.ARCH, self.DATA = load_config(self.model_path)
      with torch.no_grad():
        self.model = Segmentator(self.ARCH, len(self.DATA["learning_map_inv"]), self.model_path)
      if self.ARCH["post"]["KNN"]["use"]:
        self.post = KNN(self.ARCH["post"]["KNN"]["params"], len(self.DATA["learning_map_inv"]))

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("Infering in device: ", self.device)
      if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cudnn.benchmark = True
        cudnn.fastest = True
        self.model.eval()
        self.model.to(self.device)
      
      labels = self.DATA["labels"]
      color_map = self.DATA["color_map"]
      learning_map = self.DATA["learning_map"]
      learning_map_inv = self.DATA["learning_map_inv"]
      sensor = self.ARCH["dataset"]["sensor"]
      self.max_points = self.ARCH["dataset"]["max_points"]

      sensor_img_H = sensor["img_prop"]["height"]
      sensor_img_W = sensor["img_prop"]["width"]
      sensor_fov_up = sensor["fov_up"]
      sensor_fov_down = sensor["fov_down"]
      nclasses = len(learning_map_inv)
      self.sensor_img_means = torch.tensor(sensor["img_means"],
                                            dtype=torch.float, device=self.device)
      self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float, device=self.device)
      self.scan = LaserScan(project=True,
                      H=sensor_img_H,
                      W=sensor_img_W,
                      fov_up=sensor_fov_up,
                      fov_down=sensor_fov_down,
                      training=False)

    def preprocess(self):
      
      print(f"input points shape: {self.points.shape}")
      t = time.time()
      ### For PointCloud Preprocess
      points = self.points[:, 0:3]    # get xyz
      remissions = self.points[:, 3]  # get remission
      self.scan.set_points(points, remissions)
      print(f"part1 cost time: {time.time() - t}")
      
      unproj_n_points = self.scan.points.shape[0]
      unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float, device=self.device)
      
      unproj_range[:unproj_n_points] = torch.from_numpy(self.scan.unproj_range)
      proj_range = torch.from_numpy(self.scan.proj_range).to(self.device)
      proj_xyz = torch.from_numpy(self.scan.proj_xyz).to(self.device)
      proj_remission = torch.from_numpy(self.scan.proj_remission).to(self.device)
      proj_mask = torch.from_numpy(self.scan.proj_mask).to(self.device)

      proj_x = torch.full([self.max_points], -1, dtype=torch.long, device=self.device)
      proj_x[:unproj_n_points] = torch.from_numpy(self.scan.proj_x)
      proj_y = torch.full([self.max_points], -1, dtype=torch.long, device=self.device)
      proj_y[:unproj_n_points] = torch.from_numpy(self.scan.proj_y)
      proj = torch.cat([proj_range.unsqueeze(0), proj_xyz.permute(2, 0, 1), proj_remission.unsqueeze(0)])
      
      proj = proj * proj_mask.float()

      proj_blocked = proj.unsqueeze(1) # Swap Batch and channel dimensions

      proj = (proj - self.sensor_img_means[:, None, None]
              ) / self.sensor_img_stds[:, None, None]

      proj = proj * proj_mask.float()


      n, c, h, w = proj_blocked.size()
      windows_size = 4  # windows size
      proj_chan_group_points = f.unfold(proj_blocked, kernel_size=windows_size, stride=windows_size)

      projmask_chan_group_points = f.unfold(proj_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=windows_size,
                                            stride=windows_size)

      # Get the mean point (taking apart non-valid points
      proj_chan_group_points_sum = torch.sum(proj_chan_group_points, dim=1)
      projmask_chan_group_points_sum = torch.sum(projmask_chan_group_points, dim=1)
      proj_chan_group_points_mean = proj_chan_group_points_sum / projmask_chan_group_points_sum

      # tile it for being able to substract it to the other points
      tiled_proj_chan_group_points_mean = proj_chan_group_points_mean.unsqueeze(1).repeat(1, windows_size * windows_size,
                                                                                          1)

      # remove nans due to empty blocks
      is_nan = tiled_proj_chan_group_points_mean != tiled_proj_chan_group_points_mean
      tiled_proj_chan_group_points_mean[is_nan] = 0.

      # compute valid mask per point
      tiled_projmask_chan_group_points = (1 - projmask_chan_group_points.repeat(n, 1, 1)).byte()

      # substract mean point to points
      proj_chan_group_points_relative = proj_chan_group_points - tiled_proj_chan_group_points_mean

      # set to zero points which where non valid at the beginning
      proj_chan_group_points_relative[tiled_projmask_chan_group_points] = 0.

      # NOW proj_chan_group_points_relative HAS Xr, Yr, Zr, Rr, Dr relative to the mean point
      proj_norm_chan_group_points = f.unfold(proj.unsqueeze(1), kernel_size=windows_size, stride=windows_size)
      xyz_relative = proj_chan_group_points_relative[1:4, ...]
      
      relative_distance = torch.tensor(np.linalg.norm(xyz_relative.cpu().numpy(), ord=2, axis=0), device=self.device).float().unsqueeze(0)

      # NOW proj_norm_chan_group_points HAS X, Y, Z, R, D. Now we have to concat them both
      proj_chan_group_points_combined = torch.cat(
          [proj_norm_chan_group_points, proj_chan_group_points_relative, relative_distance], dim=0)

      # convert back to image for image-convolution-branch
      proj_out = f.fold(proj_chan_group_points_combined, proj_blocked.shape[-2:], kernel_size=windows_size,
                        stride=windows_size)
      proj_out = proj_out.squeeze(1)
      proj_out = proj_out.unsqueeze(0)
      proj_chan_group_points_combined = proj_chan_group_points_combined.unsqueeze(0)

      return proj_out, proj_mask, proj_x, proj_y, proj_range, unproj_range, proj_chan_group_points_combined

    def run(self, points):
      t_t = time.time()
      self.points = points
      print("start preprocess")
      t = time.time()
      proj_in, proj_mask, p_x, p_y, proj_range, unproj_range, proj_chan_group_points = self.preprocess()
      
      print(f"preprocess time cost: {time.time() - t} ")

      torch.cuda.synchronize()
      t = time.time()
      # compute output
      proj_output = self.model([proj_in, proj_chan_group_points], proj_mask)
      proj_argmax = proj_output[0].argmax(dim=0)
      torch.cuda.synchronize()
      print(f"network inference cost time: {time.time() - t}")

      print(f"proj_argmax {proj_argmax.shape}")

      t = time.time()
      # knn postproc
      # unproj_argmax = self.post(proj_range,
      #                           unproj_range,
      #                           proj_argmax,
      #                           p_x,
      #                           p_y)
      unproj_argmax = proj_argmax[p_y, p_x]

      pred_np = unproj_argmax.detach().cpu().numpy()
      pred_np = pred_np.reshape((-1)).astype(np.int32)
      pred_np = pred_np[:points.shape[0]]
      
      print(f"knn post-process cost time: {time.time() - t}")
      print(f"prediction shape: {pred_np.shape}")

      t = time.time() 

      pred_label_intensity = pred_np[:, np.newaxis]      
      cloud_rbg = np.concatenate((points[:, :3], pred_label_intensity), axis=1)
      print(f"output point cost time: {time.time() - t}")
      print(f"Pipeline Total Cost time: {time.time() - t_t}")
      return cloud_rbg
      
def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    t_t = time.time()
    print("")
    print("##################")
    # arr_bbox = BoundingBoxArray()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    res = proc_1.run(np_p)
    pub_cloud =  xyz_array_to_pointcloud2(res, frame_id="velodyne")
    segments_pub.publish(pub_cloud)

   
if __name__ == "__main__":

    global proc
  
    ### SemanticKITTI
    model_path = "/home/muzi2045/Documents/project/3D-MiniNet/pytorch_code/lidar-bonnetal/train/tasks/semantic/models/3D-MiniNet"

    proc_1 = Processor_ROS(model_path)
    proc_1.initialize()
    rospy.init_node('MiniNet_ros_node')
    sub_lidar_topic = [ "/velodyne_points", "/lidar_top"]
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    segments_pub = rospy.Publisher("segment_cloud", PointCloud2, queue_size=10)

    print("[+] MiniNet ROS Node has started!")    
    rospy.spin()