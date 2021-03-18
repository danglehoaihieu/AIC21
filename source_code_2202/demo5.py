
from __future__ import division, print_function, absolute_import
import sys
import os
import json
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from object_tracking import OT

from tools import generate_detections1 as gdet
from collections import deque

import platform
import shutil
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detect import run_detect
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)#plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

	
# input: start_video_id end_video_id 
#start_video = sys.argv[1]
#end_video = sys.argv[2]

data_path = '../Dataset_A'
list_video_path = '../Dataset_A/datasetA_vid_stats.txt'
id_path = '../Dataset_A/list_video_id.txt'
zones_path = '../ROIs'
video_path = '../Dataset_A'
result_path = './submission_output'
#datasetA_path = os.path.join(data_path, 'Dataset_A')
#video_path = os.path.join(datasetA_path, 'video')
#video_path = os.path.join(datasetA_path, 'short_video')
#zones_path = os.path.join(datasetA_path, 'zones1')

# Initialize
img_size=640
set_logging()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA
#half= False

# Load model
model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
if half:
    model.half() 
img = torch.zeros((1, 3, img_size, img_size), device=device)
_ = model(img.half() if half else img) if device.type != 'cpu' else None

def draw_roi(roi, frame):
    roi_nums = len(roi)
    for i in range(roi_nums):
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,0),2)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,0),2)
    return frame
def load_roi(rois_path, name_video):
    roi = []
    cam_index = name_video.split('_')[1]
    if len(cam_index) > 2:
        cam_index = cam_index.split('.')[0]
    print(cam_index)
    with open(os.path.join(rois_path, 'cam_{}.txt'.format(cam_index))) as f:
        roi = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in f]
    return roi

def load_list_video(input_path, id_path):
    names = []
    ids = []
    info = []
    with open(id_path,'r') as f:
        for line in f:
            a = line.split(' ')
            ids.append(a[0])
            names.append(a[-1].split('\n')[0])

    with open(input_path,'r') as f:
        for line in f:
            video_name = line.split('\t')[0]

            try:
                fps = line.split('\t')[1]
                if fps != 'fps':
                    fps = fps.split('/')[:-1]
                    fps = int(fps[0])
                id = ids[names.index(video_name)]
                info.append([id, video_name, fps])
            except:
                pass
    print(info)
    return info
def load_dets(dets_path, size_img):
    dets = []
    cars1frame = []
    trucks1frame = []
    current_id = 1
    def append_det(clas, cx, cy, w, h, conf):
        if clas == 2:
            cars1frame.append([cx-w//2, cy-h//2, w, h, conf])
        else:
            trucks1frame.append([cx-w//2, cy-h//2, w, h, conf])
    with open(dets_path,'r') as f:
        for line in f:
            info = line.split(' ')
            frame_id = int(info[0])
            clas = int(info[1])
            cx = int(size_img[0]*float(info[2]))
            cy = int(size_img[1]*float(info[3]))
            w = int(size_img[0]*float(info[4]))
            h = int(size_img[1]*float(info[5]))
            conf = float(info[6])
            if frame_id != current_id:
                dets.append([cars1frame, trucks1frame])
                cars1frame = []
                trucks1frame = []
                current_id = frame_id
            append_det(clas, cx, cy, w, h, conf)

    return dets

def main():
    re = load_dets('../Dataset_A/cam_7_dawn.txt', size_img=(2*640, 2*480))
    #print(re[0])
    #return 0
    counter = []
    writeVideo_flag = True
    fps = 0.0
    filename_path = os.path.join(result_path, 'submission.txt')
    info_cam = load_list_video(list_video_path, id_path)
    result_file = open(filename_path, 'w')

    max_cosine_distance=0.8
    nn_budget = 100
    nms_max_overlap = 1.0
    display = True
    for info in info_cam: 
        path = os.path.join(video_path, info[1])
        ROI = load_roi(zones_path, info[1])

        car_class = OT(1,'Car')
        truck_class = OT(1,'Truck')
        objects = [car_class, truck_class]
        print("Processing video: ", info)
        video_capture = cv2.VideoCapture(path)
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('result_'+info[1]+'.avi', fourcc, info[-1], (w, h))

        pause_display = False
        frame_num = 0
        data = []
        start_video = time.time()
        while True:
            
            start = time.time()
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video_capture.read()  # frame shape 640*480*3
            
            
            if ret != True:
                break
             #   print(count1)
            print(frame.shape)
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))   
            result = []
            t1 = time.time()
            res = frame.copy()

            dets = re[frame_num]
            
            for det, ob in zip (dets, objects):
                    ob.predict_obtracker(frame, det)
                    ob.update_obtracker()
                    frame = ob.visualize(frame)
                    res, data = ob.tracking_ob1(ROI, frame, info[0],frame_num, data)

            draw_roi(ROI, res)

            if writeVideo_flag:
                    #save a frame
                   out.write(frame)
            frame_num += 1
            '''
            cv2.imshow('frame', res)
            
            if not pause_display:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                if key == ord(' '):
                    pause_display = not pause_display
                frame_num += 1
            else:
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                if key == ord(' '):
                    pause_display = not pause_display
            '''
            print('frame_num: {} fps: {} '.format(frame_num, (1/(time.time()-start))))
        print('Process'+ info[0]+' :{} h', format((time.time()-start_video)/60))
    print(" ")
    print("[Finish]")
        

    video_capture.release()

    if writeVideo_flag:
        out.release()
        #list_file.close()
    result_file.close()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
