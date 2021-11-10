'''
Description: Do not Edit
Author: muzi2045
Date: 2021-03-01 17:27:28
LastEditor: muzi2045
LastEditTime: 2021-03-03 16:10:54
'''

import open3d as o3d

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

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

def init_model(model_path : str):

  ARCH, DATA = load_config(model_path)

  with torch.no_grad():
    model = Segmentator(ARCH, len(DATA["learning_map_inv"]), model_path)
  
  if ARCH["post"]["KNN"]["use"]:
    post = KNN(ARCH["post"]["KNN"]["params"], len(DATA["learning_map_inv"]))

  # GPU?
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Infering in device: ", device)
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    cudnn.benchmark = True
    cudnn.fastest = True
    model.eval()
    model.to(device)
  return model, post

def preprocess(model_path, input):

  print(f"input points shape: {input.shape}")
  ### For Read Config
  ARCH, DATA = load_config(model_path)
  labels = DATA["labels"]
  color_map = DATA["color_map"]
  learning_map = DATA["learning_map"]
  learning_map_inv = DATA["learning_map_inv"]
  sensor = ARCH["dataset"]["sensor"]
  max_points = ARCH["dataset"]["max_points"]

  sensor_img_H = sensor["img_prop"]["height"]
  sensor_img_W = sensor["img_prop"]["width"]
  sensor_fov_up = sensor["fov_up"]
  sensor_fov_down = sensor["fov_down"]
  nclasses = len(learning_map_inv)
  sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
  sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
  scan = LaserScan(project=True,
                  H=sensor_img_H,
                  W=sensor_img_W,
                  fov_up=sensor_fov_up,
                  fov_down=sensor_fov_down,
                  training=False)
  ### For PointCloud Preprocess
  points = input[:, 0:3]    # get xyz
  remissions = input[:, 3]  # get remission
  scan.set_points(points, remissions)
  
  unproj_n_points = scan.points.shape[0]
  unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
  unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
  unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
  unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
  unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
  unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
  unproj_labels = []

  proj_range = torch.from_numpy(scan.proj_range).clone()
  proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
  proj_remission = torch.from_numpy(scan.proj_remission).clone()
  proj_mask = torch.from_numpy(scan.proj_mask)
  proj_labels = []

  proj_x = torch.full([max_points], -1, dtype=torch.long)
  proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
  proj_y = torch.full([max_points], -1, dtype=torch.long)
  proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
  proj = torch.cat([proj_range.unsqueeze(0).clone(),
                    proj_xyz.clone().permute(2, 0, 1),
                    proj_remission.unsqueeze(0).clone()])

  proj = proj * proj_mask.float()

  proj_blocked = proj.unsqueeze(1) # Swap Batch and channel dimensions

  proj = (proj - sensor_img_means[:, None, None]
          ) / sensor_img_stds[:, None, None]

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
  relative_distance = torch.tensor(np.linalg.norm(xyz_relative.numpy(), ord=2, axis=0)).float().unsqueeze(0)

  # NOW proj_norm_chan_group_points HAS X, Y, Z, R, D. Now we have to concat them both
  proj_chan_group_points_combined = torch.cat(
      [proj_norm_chan_group_points, proj_chan_group_points_relative, relative_distance], dim=0)

  # convert back to image for image-convolution-branch
  proj_out = f.fold(proj_chan_group_points_combined, proj_blocked.shape[-2:], kernel_size=windows_size,
                    stride=windows_size)
  proj_out = proj_out.squeeze(1)
  proj_out = proj_out.unsqueeze(0)
  proj_chan_group_points_combined = proj_chan_group_points_combined.unsqueeze(0)
  return proj_out, proj_mask, proj_x, proj_y, proj_range, unproj_range, unproj_n_points, proj_chan_group_points_combined
  

def inference(model_path : str, points):
  ARCH, DATA = load_config(model_path)
  print("init model")

  model, post = init_model(model_path)
  print("start preprocess")  
  proj_in, proj_mask, p_x, p_y, proj_range, unproj_range, npoints, proj_chan_group_points = preprocess(model_path, points)

  print(f"proj_in shape: {proj_in.shape}")
  print(f"proj_chan_group_points shape: {proj_chan_group_points.shape}")
  proj_in = proj_in.cuda()
  proj_chan_group_points = proj_chan_group_points.cuda()
  proj_mask = proj_mask.cuda()
  p_x = p_x.cuda()
  p_y = p_y.cuda()
  proj_range = proj_range.cuda()
  unproj_range = unproj_range.cuda()

  # compute output
  proj_output = model([proj_in, proj_chan_group_points], proj_mask)

  proj_argmax = proj_output[0].argmax(dim=0)

  print(f"proj_argmax {proj_argmax.shape}")

    # knn postproc
  unproj_argmax = post(proj_range,
                            unproj_range,
                            proj_argmax,
                            p_x,
                            p_y)

  pred_np = unproj_argmax.detach().cpu().numpy()
  pred_np = pred_np.reshape((-1)).astype(np.int32)
  pred_np = pred_np[:points.shape[0]]
  
  print(f"prediction shape: {pred_np.shape}")

  learning_map_inv = DATA['learning_map_inv']
  color_map = DATA['color_map']

  pred_label_shift = np.zeros(pred_np.shape[0], dtype=np.uint16)
  pred_label_color = np.zeros((pred_np.shape[0], 3), dtype=np.uint8)

  for i in range(pred_np.shape[0]):
    pred_label_shift[i] = learning_map_inv[pred_np[i]]

  for i in range(pred_np.shape[0]):
    pred_label_color[i] = color_map[pred_label_shift[i]][::-1]

  cloud_rbg = np.concatenate((points[:, :3], pred_label_color), axis=1)
  return cloud_rbg


if __name__ == "__main__":

  pc_path = "/media/muzi2045/416d169d-8634-4f5c-be81-9cf0df1bb87b/Datatset/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin"
  model_path = "/home/muzi2045/Documents/project/3D-MiniNet/pytorch_code/lidar-bonnetal/train/tasks/semantic/models/3D-MiniNet"

  point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
  
  print("######## start inference #############")
  output = inference(model_path, point_cloud)
  print("######## end inference ###############")
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(output[:, :3])
  pcd.colors = o3d.utility.Vector3dVector(output[:, 3:6])
  o3d.visualization.draw_geometries([pcd])
  

  
