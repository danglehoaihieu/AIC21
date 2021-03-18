
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
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
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
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
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

data_path = './test_data'
video_path = data_path
zones_path = './zones_team10'
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
def load_zone_anno(video_id, json_filename):
    if video_id > 9:
        json_filename = os.path.join(json_filename, 'cam_{}.json'.format(video_id))
    else:
        json_filename = os.path.join(json_filename, 'cam_0{}.json'.format(video_id))
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
        roi_expand = [(int(x), int(y)) for x, y in dd['shapes'][1]['points']]
        paths = {}
        for it in dd['shapes'][2:]:
            kk = str(int(it['label'][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y in it['points']]
        
    return polygon, paths, roi_expand
def cut_roi(image, roi):
    mask_roi = np.zeros((image.shape), np.uint8)
    roi = np.array([roi], dtype=np.int32)
    mask_roi = cv2.fillPoly(mask_roi, roi, (255, 255, 255))
    #mask_roi = cv2.fillPoly(mask_roi, roi, (0, 0, 0))
    image = cv2.bitwise_and(image, mask_roi) 
    return image       
def import_videos(ids, video_dir=video_path):
    vlist = []
    if ids > 9:
        video = os.path.join(video_dir, 'cam_{}.mp4'.format(ids))
        return video
    else:
        video = os.path.join(video_dir, 'cam_0{}.mp4'.format(ids))
        return video
        
def draw_roi(roi, frame):
    roi_nums = len(roi)
    for i in range(roi_nums):
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,255),4)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,255),4)
    return frame
def load_list_IDcams(directory):
    list_ids = []
    for root,dirs,files in os.walk(directory):
        for file_ in files:
            if file_.endswith(".json"):
                name = file_.split('.')[0]
                index = name.split('_')[1]
                list_ids.append(int(index))
    print("list", list_ids)
    return list_ids    
def write_result(video_id, moi_nums, result_file, data):
    if moi_nums > 2 or video_id == 16 or video_id == 17:
        for d in data:
            if d[0]>9:
                result_file.write('cam_{} {} {} {}\n'.format(d[0], d[1], d[2], d[3]))
            else:
                result_file.write('cam_0{} {} {} {}\n'.format(d[0], d[1], d[2], d[3]))
    else:
        delay = np.zeros((4, moi_nums,2))
        for d in data:
            if not d.flag_cut_roi:
                continue
            delay[d.id_class-1, int(d.direc)-1, 0] += d.frame_delay # sum frame
            delay[d.id_class-1, int(d.direc)-1, 1] += 1 # counter
        for d in data:
            if  video_id==4 or video_id==5:
                if d.direc =='2': 
                    d.frame_cut_moi += 5
                else:
                    d.frame_cut_moi += 4

            elif video_id == 2 or video_id==3:
                if d.direc == '1':
               	    if d.id_class==1:
                    	d.frame_cut_moi += 5
                else:
                    if d.id_class ==2:
                    	d.frame_cut_moi += 5
                    elif d.id_class ==3 or d.id_class==4:
                    	d.frame_cut_moi += 7
                    #d.frame_cut_moi = d.frame_cut_moi + 14 if d.id_class == 1 else (d.frame_cut_moi + 30)
            elif video_id ==17 or video_id ==16:
                if d.direc == '1':
                    d.frame_cut_moi += 4
                else: 
                    d.frame_cut_moi += 2
            elif video_id == 9:
            	if d.id_class == 3 or d.id_class==4:
                	d.frame_cut_moi += 3
            elif video_id == 6:
                if d.direc =='2':
                	if d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi += 7
                	elif d.id_class ==2:
                		d.frame_cut_moi += 2
                else:
                	if d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi += 3
                	elif d.id_class ==2:
                		d.frame_cut_moi += 1
            elif video_id == 7 or video_id==8:
                if d.direc =='2':
                	if d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi += 3
            elif video_id == 11:
            	d.frame_cut_moi += 1
            elif video_id == 12:
            	d.frame_cut_moi += 1
            	if d.id_class ==2:
            		d.frame_cut_moi += 1
            	elif d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi +=2
            elif video_id == 13:
                if d.direc =='2':
                	if d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi += 6
                	elif d.id_class == 2:
                		d.frame_cut_moi += 3
                	else:
                		d.frame_cut_moi += 1
                else:
                	if d.id_class == 3 or d.id_class==4:
                		d.frame_cut_moi += 3
            elif video_id == 18 or video_id == 19:
            	if d.direc=='2':
            		if d.id_class == 1:
            			d.frame_cut_moi += 1
            		else:
            			d.frame_cut_moi += 3
            	else:
            		if d.id_class != 1:
            			d.frame_cut_moi += 2

           #	frame_id = frame_id + 6
            else:
                d.frame_cut_moi += 2
            fram_id = 0
            if d.flag_cut_roi:
                frame_id = d.frame_cut_moi+d.frame_delay
            else:
                Sum = delay[d.id_class-1, int(d.direc)-1, 0]
                count = delay[d.id_class-1, int(d.direc)-1, 1]
                if count == 0:
                    Sum = 5
                    count = 1
                frame_id = d.frame_cut_moi + int(Sum/count)
                
            if video_id > 9:
                result_file.write('cam_{} {} {} {}\n'.format(video_id, frame_id, d.direc, d.id_class))
            else:
                result_file.write('cam_0{} {} {} {}\n'.format(video_id, frame_id, d.direc, d.id_class))
        
            
            
def main():
    start = time.time()
    counter = []
    writeVideo_flag = True
    fps = 0.0
    filename_path = os.path.join(result_path, 'submission.txt')
    list_ids = load_list_IDcams(data_path)
    result_file = open(filename_path, 'w')
    for video_id in list_ids:
        print("video_id: %s" % str(video_id))    
        video_name = import_videos(video_id)
        print("Processing video: ",video_name)
        motor_class = thor(1,'motor')
        car_class = thor(2,'car')
        truck_class = thor(4,'truck')
        bus_class = thor(3,'bus')
        class_obstacle = [motor_class, car_class, bus_class, truck_class]
        video_capture = cv2.VideoCapture(video_name)
        #get ROT and MOT from .json file
        ROI, MOIS, roi_expand = load_zone_anno(video_id, zones_path)
        lines_roi = []
        for a, b in MOIS.items():
            lines_roi.append(b)
        moi_nums = len(MOIS)
        roi_nums = len(ROI)
        counter = np.zeros(moi_nums, dtype=int) 
        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #out = cv2.VideoWriter('./output/output_15.avi', fourcc, 30, (w, h))
            #list_file = open('detection_rslt.txt', 'w')
            frame_index = -1
        count=0
        count1=0
        data = []
            
        frame_index = 0
        j = 0
        while True:
            
            start = time.time()
           # print(count)
            ret, frame = video_capture.read()  # frame shape 640*480*3
            # gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # frame = np.zeros_like(frame1)
            # frame[:,:,0] = gray
            # frame[:,:,1] = gray
            # frame[:,:,2] = gray
            if ret != True:
                write_result(video_id, moi_nums, result_file, data)
                break
            count += 1 
            frame_index += 1
            if count == 3 or count == 2:
                if count == 3:
                    count = 0
                print(frame_index)
                 #   print(count1)
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))   
                frame = cut_roi(frame, roi_expand)
                result = []
                t1 = time.time()
                img = letterbox(frame, new_shape=img_size)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                dets = run_detect(model,img,device,frame)
                img = np.zeros((h, w, 3), np.uint8)
                img = frame.copy()
                for det, class_ob in zip (dets, class_obstacle):
                    if len(det) == 0:	
                        result.append([])
                        continue
                    boxes = [box[:4] for box in det]
                    class_ob.predict_obtracker(img, boxes)
                    class_ob.update_obtracker()
                    if moi_nums <= 2 and video_id != 16 and video_id != 17:
                        frame, data =class_ob.tracking_ob1(ROI,MOIS, frame,video_id,frame_index, data)
                    else:
                        frame, data =class_ob.tracking_ob2(ROI,MOIS, frame,video_id,frame_index, data)  
             
                frame = draw_roi(ROI, frame)
                #if moi_nums <=2:
                #    for num, line in MOIS.items():
                #        cv2.line(frame,line[2],line[3],(0,255,255),4)
                #cv2.namedWindow("YOLO4_Deep_SORT", 0);
                #cv2.resizeWindow('YOLO4_Deep_SORT', 768, 576);
                #cv2.imshow('YOLO4_Deep_SORT', frame)
                
                if writeVideo_flag:
                    #save a frame
                   out.write(frame)
                #key = cv2.waitKey(0)
                #if key & 0xFF == ord('q'):
                #     write_result(video_id, moi_nums, result_file, data)
                #     break
                if moi_nums <= 2 and video_id != 16 and video_id != 17:
                    print("data:")
                    for d in data:
                        print(d.save_data)
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
