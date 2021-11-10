'''
Description: Do not Edit
Author: muzi2045
Date: 2021-02-26 17:12:27
LastEditor: muzi2045
LastEditTime: 2021-03-03 11:11:23
'''

import os
import numpy as np
import torch
import yaml

from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.dataset.kitti.parser import Parser

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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Infering in device: ", device)
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    model.eval()
    model.to(device)
  return model, post

def onnx_export(model_path: str, save_path=None):
  ARCH, DATA = load_config(model_path)
  model, post = init_model(model_path)
  
  print("Starting export onnx model.")
  dinput = torch.randn(1, 11, 64, 2048, device='cuda')
  input_points = torch.randn(1, 11, 16, 8192, device='cuda')

  input_names = ["proj_in", "proj_points"]
  inputs = [[dinput, input_points]]
  # inputs = [[proj_in, proj_chan_group_points]]
  inputs = tuple(inputs)

  torch.onnx.export(model, inputs, save_path + "/mininet3d_test.onnx", verbose=True,
                    input_names=input_names, export_params=True, keep_initializers_as_inputs=True, opset_version=11)

if __name__ == "__main__":
    model_path = "/home/muzi2045/Documents/project/3D-MiniNet/pytorch_code/lidar-bonnetal/train/tasks/semantic/models/3D-MiniNet"
    onnx_save_path = "/home/muzi2045/Documents/project/3D-MiniNet/pytorch_code/lidar-bonnetal/train/tasks/semantic/onnx_export"
    
    print("######## start onnx export #############")
    output = onnx_export(model_path, save_path=onnx_save_path)
    print("########  end  onnx export ###############")