from __future__ import division, print_function, absolute_import
import os
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
from tools import generate_detections1 as gdet
from application_util import visualization
from deep_sort.detection import Detection as ddet
from collections import deque
from check_moi import*
import math

np.random.seed(1)
class OT(object):
    
    def __init__(self, class_id, names):
        self.max_cosine_distance = 0.9
        self.nms_max_overlap = 0.5
        self.nn_budget = None
        self.model_filename = 'model_data/market1501.pb'
        self.detections = []
        self.id = class_id
        self.boxes_tracked = []
        self.color = (255, 255, 255)
        #self.encoder = gdet.create_box_encoder(self.model_filename,batch_size=1)
        self.class_names = names
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.obtracker = Tracker(metric)
        self.pts = [deque(maxlen=100) for _ in range(999999)]
        self.point_first = [deque(maxlen=2) for _ in range(999999)]
        self.vis=visualization.Visualization(img_shape=(960,1280,3), update_ms=2000)
        COLORS = np.random.randint(100, 255, size=(255, 3), dtype="uint8")
        self.color = [int(c) for c in COLORS[self.id]]
        
    def predict_obtracker(self, frame, dets):

        boxs = [d[:4] for d in dets]
        #features = gdet.HOG_feature(frame, boxs)
        features = gdet.create_his(frame, boxs)
        self.detections = [Detection(det[:4], det[4], f) for det, f in zip(dets, features)]
        self.obtracker.predict()
    
    def update_obtracker(self):
        self.obtracker.update(self.detections)
    def remove_track(self, ids_del):
        for track in self.obtracker.tracks:
            if track.track_id == ids_del:
                self.obtracker.tracks.remove(track)
                break
    def tracking_ob1(self, ROI, MOI, frame, video_id, frame_id, data, frame_delay):
        ci = 0
        indexIDs = []
        c = []
        self.boxes_tracked = []
        for track in self.obtracker.tracks:
            # if (track.time_since_update > 1):#not track.is_confirmed() or 

            #     continue
            ci+=1
            bbox = track.to_tlbr()
            color = self.color
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            
            self.pts[track.track_id].append(center)
            #center point
            #cv2.circle(frame,  (center), 1, (255,0, 0), thicknes
            l = len(self.pts[track.track_id])
            if l < 10: continue
            p0 = self.pts[track.track_id][0]
            p1 = self.pts[track.track_id][int(l/3)]
            p2 = self.pts[track.track_id][int(l*2/3)]
            p3 = self.pts[track.track_id][l-1]
            vel = math.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)
            if not track.out_roi and vel >= 10:
                try:
                    direc = predict_direction(ROI, [p0,p1,p2,p3], MOI)
                except:
                    print('bug predict direc')
                    direc = None
                    pass
                if direc != None:
                    track.out_roi = True
                    frame = frame_id + frame_delay[int(self.id)-1][int(direc-1)]
                    data.append([video_id, frame, direc, self.id])
            
            cv2.arrowedLine(frame,p0,p1,(color),2)
            cv2.arrowedLine(frame,p1,p2,(color),2)
            cv2.arrowedLine(frame,p2,p3,(color),2)
            #cv2.putText(frame, str(self.class_names),(int(bbox[0]), int(bbox[1]-5)),0, 5e-3 * 100, (color),2)

            #print('track {}, mean {}, covariance {}'. format(track.track_id, track.mean, track.covariance))
        
        return frame, data
    def visualize(self,frame):
        self.vis.set_image(frame.copy())
        self.vis.draw_detections(self.detections)
        self.vis.draw_trackers(self.obtracker.tracks)
        return self.vis.return_img()
        
    
    
    
    
    
    
    
        
        
        
