
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
from application_util import visualization
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


pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
	
	
	
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
    id_list = []
    video_list = []
    names = []
    ids = []
    with open(id_path,'r') as f:
        for line in f:
            a = line.split(' ')
            ids.append(a[0])
            names.append(a[-1].split('\n')[0])

    with open(input_path,'r') as f:
        for line in f:
            video_name = line.split('\t')[0]
            try:
                id = ids[names.index(video_name)]
                id_list.append(id)
                video_list.append(video_name)
            except:
                pass
        print(id_list)
        print(video_list)
    return video_list, id_list

def main():
    start = time.time()
    counter = []
    writeVideo_flag = False
    fps = 0.0
    filename_path = os.path.join(result_path, 'submission.txt')
    list_video, list_ids = load_list_video(list_video_path, id_path)
    result_file = open(filename_path, 'w')

    max_cosine_distance=0.8
    nn_budget = 100
    nms_max_overlap = 1.0
    display = True
    for video in list_video: 
        path = os.path.join(video_path, video)
        ROI = load_roi(zones_path, video)
        vis=visualization.Visualization(img_shape=(960,1280,3), update_ms=2000)

        metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        results = []
        print("Processing video: ", video )
        video_capture = cv2.VideoCapture(path)

        pause_display = False
        frame_num = 0
        while True:
            
            start = time.time()
            # print(count)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video_capture.read()  # frame shape 640*480*3
            # gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # frame = np.zeros_like(frame1)
            # frame[:,:,0] = gray
            # frame[:,:,1] = gray
            # frame[:,:,2] = gray
            
            if ret != True:
                break
             #   print(count1)
            #print(frame.shape)
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))   
            result = []
            t1 = time.time()
            
            img = letterbox(frame, new_shape=img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            dets = run_detect(model,img,device,frame)

            detectionss=[]
            for det in dets:
                feature = gdet.HOG_feature(frame, det[:4])
                detectionss.append(Detection(det[:4], det[4], feature, det[-1]))   
            #detectionss.append(Detection(det[:4], det[4], det[-1]) for det in dets)
            img = np.zeros((h, w, 3), np.uint8)
            img = frame.copy()
            min_confidence = 0.4
            detections = [d for d in detectionss if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            if display:
                vis.set_image(frame.copy())
                vis.draw_detections(detections)
                vis.draw_trackers(tracker.tracks)
            res = vis.return_img()
            draw_roi(ROI, res)
            cv2.imshow('frame', res)
            print('frame_num', frame_num)
            if not pause_display:
                key = cv2.waitKey(2)
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
