
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
mois_path = '../movement_description'

filename_path = os.path.join(result_path, 'submission.csv')
result_file = open(filename_path, 'w')

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
    roi_nums = len(roi)-1
    for i in range(roi_nums):
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,0),2)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,0),2)
    return frame
def load_roi_moi(rois_path, mois_path, name_video):
    roi = []
    mois = {}
    list_moi_edge=[]
    cam_index = name_video.split('_')[1]
    if len(cam_index) > 2:
        cam_index = cam_index.split('.')[0]
    print(cam_index)
    with open(os.path.join(rois_path, 'cam_{}.txt'.format(cam_index))) as f:
    	for p in f:
    		p = p.rstrip("\n")
    		p = p.split(',')
    		temp = p[2:]
    		temp = [int(x) for x in temp]
    		list_moi_edge.append(temp)
    		roi.append((int(p[0]), int(p[1])))
    roi.append(list_moi_edge)
    with open(os.path.join(mois_path, 'cam_{}.txt'.format(cam_index))) as f:
    	for i, line in enumerate(f):
    		line = line.rstrip("\n")
    		if len(line) == 0: continue
    		a = line.split(',')
    		p1 = (int(a[0]),int(a[1]))
    		p2 = (int(a[2]),int(a[3]))
    		p3 = (int(a[4]),int(a[5]))
    		p4 = (int(a[6]),int(a[7]))
    		l1 = (int(a[8]),int(a[9]))
    		l2 = (int(a[10]),int(a[11]))
    		mois[i+1]=[p1,p2,p3,p4,l1,l2]
    return roi, mois
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
def load_delay(path, name_video):
    f = open(path,'r').readlines()
    posi = 0
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if line == name_video:
            posi = i
            break
    car = [int(t) for t in f[posi+1].rstrip('\n').split(',')]
    truck = [int(t) for t in f[posi+2].rstrip('\n').split(',')]
    return [car, truck]
def write_result_file(data):
    for d in data:
        result_file.write('{},{},{},{}\n'.format(d[0], d[1], d[2], d[3]))
def main():
    counter = []
    writeVideo_flag =False
    display = False
    fps = 0.0
    info_cam = load_list_video(list_video_path, id_path)
    

    max_cosine_distance=0.8
    nn_budget = 100
    nms_max_overlap = 1.0
    for info in info_cam: 
        path = os.path.join(video_path, info[1])
        ROI, MOI = load_roi_moi(zones_path, mois_path, info[1])
        frame_delay = load_delay('../Dataset_A/time_delay.txt', info[1])
        print('delay', frame_delay)
        print('MOI', MOI)
        car_class = OT(1,'Car')
        truck_class = OT(2,'Truck')
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
        frame_num = 17900
        data = []
        start_video = time.time()
        while True:
            
            start = time.time()
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video_capture.read()  # frame shape 640*480*3
            
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
            res = frame.copy()
            dets = run_detect(model,img,device,frame)
            
            for det, ob in zip (dets, objects):
                    ob.predict_obtracker(frame, det)
                    ob.update_obtracker()
                    #frame = ob.visualize(frame)
                    res, data = ob.tracking_ob1(ROI, MOI, frame, info[0],frame_num, data, frame_delay)
            print('saved', len(data))
            
            
            #frame_num += 1
            if display:
                draw_roi(ROI, res)
                cv2.imshow('frame', res)
                if not pause_display:
                    key = cv2.waitKey(10)
                    if key == ord('q'):
                        break
                    if key == ord(' '):
                        pause_display = not pause_display
                    frame_num += 1
                else:
                    key = cv2.waitKey(10)
                    if key == ord('q'):
                        break
                    if key == ord(' '):
                        pause_display = not pause_display
            if writeVideo_flag:
                #save a frame
                out.write(frame)
            print('frame_num: {} fps: {} '.format(frame_num, (1/(time.time()-start))))
        print('Process'+ info[0]+' :{} h', format((time.time()-start_video)/60))
        write_result_file(data)
        print('wrote')
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
