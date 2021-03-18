import os
import cv2
import numpy as np

data_path = '../Dataset_A'
list_video_path = '../Dataset_A/datasetA_vid_stats.txt'
id_path = '../Dataset_A/list_video_id.txt'
zones_path = '../ROIs'
video_path = '../Dataset_A'
result_path = './submission_output'
mois_path = '../movement_description'

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
def draw_roi(roi, frame):
    roi_nums = len(roi)-1
    for i in range(roi_nums):
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,0),2)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,0),2)
    return frame	
def main():
	ROI, MOI = load_roi_moi(zones_path, mois_path, 'cam_2.mp4')
	print('mmoi',MOI)
	print('roi',ROI[-1])
	lines = ROI[:-1]
	direc_proposed = [1,3,10]
	moi_proposed = [p for d, p in MOI.items() if d in direc_proposed]
	a = moi_proposed[0][0:2]
	print(a[0])
	for i,p in enumerate(zip(lines, lines[1:]+lines[:1])):
		print(i, p[0], p[1])
	img = cv2.imread('/home/hiu/Downloads/AIC21_Track1_123/screen_shot_with_roi_and_movement/cam_2.jpg')
	# for num, line in MOI.items():
	# 	for i in range(len(line)-1):
	# 		cv2.line(img,line[i],line[i+1],(0,255,0),2)
	img = draw_roi(ROI, img)
	res = cv2.resize(img, (int(0.8*img.shape[1]), int(0.8*img.shape[0])), interpolation = cv2.INTER_AREA)
	cv2.imshow('res', res)
	cv2.waitKey(0)
if __name__ == '__main__':
	main()