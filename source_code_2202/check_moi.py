import cv2
import numpy as np
import math


# Idea:  
# 1) Draw a horizontal line to the right of each point and extend it to infinity

# 2) Count the number of times the line intersects with polygon edges.

# 3) A point is inside the polygon if either count of intersections is odd or
#    point lies on an edge of polygon.  If none of the conditions is true, then 
#    point lies outside.

# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True 
    return False 


# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  
  	# colinear 
    if (val == 0):
    	return 0  			

   	# clock or counterclock wise 
    if (val > 0):
    	return 1
    else:
    	return 2

def is_intersect(p1, q1, p2, q2):
	# Find the four orientations needed for general and special cases 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if (o1 != o2 and o3 != o4):
        return True 
  
    # Special Cases 
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    if (o1 == 0 and onSegment(p1, p2, q1)):
    	return True
  
    # p1, q1 and p2 are colinear and q2 lies on segment p1q1 
    if (o2 == 0 and onSegment(p1, q2, q1)):
    	return True
  
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    if (o3 == 0 and onSegment(p2, p1, q2)):
    	return True 
  
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    if (o4 == 0 and onSegment(p2, q1, q2)):
    	return True
  
    return False # Doesn't fall in any of the above cases

def is_point_in_polygon(polygon, point):
	# Create a point for line segment from p to infinite 
	extreme = [point[0], 1e9]

	# Count intersections of the above line with sides of polygon 
	count = 0
	i = 0

	while True:
		j = (i+1) % len(polygon)

		# Check if the line segment from 'p' to 'extreme' intersects 
        # with the line segment from 'polygon[i]' to 'polygon[j]'
		if is_intersect(polygon[i], polygon[j], point, extreme):
			# If the point 'p' is colinear with line segment 'i-j', 
			# then check if it lies on segment. If it lies, return true, 
			# otherwise false 
			if orientation(polygon[i], point, polygon[j])==0:
				return onSegment(polygon[i], point, polygon[j])
			count = count + 1

		i = j
		if i==0:
			break
	
	return count % 2 == 1

# use this function to check if a bounding box is inside the polygon 
def is_bounding_box_intersect(bounding_box, polygon):
	for i in range(len(bounding_box)):
		if is_point_in_polygon(polygon, bounding_box[i]):
			return True
	return False

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def cosine(line, orbi):
    u = (line[1][0]-line[0][0], line[1][1]-line[0][1])
    v = (orbi[1][0]-orbi[0][0], orbi[1][1]-orbi[0][1])
    c = (u[0]*v[0]+u[1]*v[1])/math.sqrt(u[0]**2+u[1]**2)/math.sqrt(v[0]**2+v[1]**2)
    return c
def matrix_similarity(MOI, p, i):
    array = np.ones(MOI.shape[0])
    for k in range(MOI.shape[0]):
        array[k] = cosine(MOI[k][i:i+2], p[i:i+2])
    return array
def index_line_intersect(orbi_2, orbi_3, lines):
    for i, p in enumerate(zip(lines, lines[1:]+lines[:1])):
        if is_intersect(p[0],p[1], orbi_2, orbi_3):
            return i    
    return None
def predict_direction(ROI, orbi, MOI):
    if is_point_in_polygon(ROI[:-1], orbi[-2]) and not is_point_in_polygon(ROI[:-1], orbi[-1]):
        index = index_line_intersect(orbi[-2], orbi[-1], ROI[:-1])
        if index == None: return None
        direc_proposed = ROI[-1][index]
        moi_proposed = np.array([p[:-2] for d, p in MOI.items() if d in direc_proposed])
        if len(direc_proposed) == 1:
            return direc_proposed[0]
        else:
            cosin_array = np.zeros(len(direc_proposed))
            for i in range(len(orbi)-1):
                cosin_array += matrix_similarity(moi_proposed, orbi, i)
                #print('cosin_array', cosin_array)
            return direc_proposed[np.argmax(cosin_array)]
    else:
        return None
