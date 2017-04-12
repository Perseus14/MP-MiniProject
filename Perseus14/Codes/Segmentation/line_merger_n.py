import math
import cv2
from matplotlib import pyplot as plt
from matplotlib import pylab
import pickle

file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)

def length ( line ):
    #returns length of line segment
    x1 = float(line[0])
    x2 = float(line[2])
    y1 = float(line[1])
    y2 = float(line[3])
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    
   
def angle ( line ): 
    #returns angle of line in degrees
    #calculation does not consider shifted top left origin as shift is
    #applicable to all lines
    x1 = float(line[0])
    x2 = float(line[2])
    y1 = float(line[1])
    y2 = float(line[3])
    if(x2 == x1):
        if (y2 > y1):
            return 90
        else:
            return 270                
    #range is [0, 360] degrees
    return 180 + math.degrees(math.atan2((y2 - y1), (x2 - x1)))    


def isHorizontallyClose( line_segment1, line_segment2, spatial_proximity_threshold ):
    x11 = float(line_segment1[0])
    x12 = float(line_segment1[2])   
    x21 = float(line_segment2[0])
    x22 = float(line_segment2[2])
    return (abs(x11 - x21) < spatial_proximity_threshold or 
        abs(x11 - x22) < spatial_proximity_threshold or
        abs(x12 - x21) < spatial_proximity_threshold or
        abs(x12 - x22) < spatial_proximity_threshold)
    

def isVerticallyClose( line_segment1, line_segment2, spatial_proximity_threshold ):
    y11 = float(line_segment1[1])
    y12 = float(line_segment1[3])    
    y21 = float(line_segment2[1])
    y22 = float(line_segment2[3])
    return (abs(y11 - y21) < spatial_proximity_threshold or 
        abs(y11 - y22) < spatial_proximity_threshold or
        abs(y12 - y21) < spatial_proximity_threshold or
        abs(y12 - y22) < spatial_proximity_threshold)    


def distanceBetweenTwoPoints(x1, y1, x2, y2):
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    
def mergeTwoLines ( line_segment1, line_segment2, spatial_distance_parameter, angular_difference_threshold ) :
    l1 = length(line_segment1)
    l2 = length(line_segment2)
    if(l1 < l2):
        print "This should not happen since sorted by length"        
    point1_x = float(line_segment1[0]) 
    point1_y = float(line_segment1[1])
    point2_x = float(line_segment1[2])
    point2_y = float(line_segment1[3])
    point3_x = float(line_segment2[0])
    point3_y = float(line_segment2[1])
    point4_x = float(line_segment2[2])
    point4_y = float(line_segment2[3])
    d1 = distanceBetweenTwoPoints(point1_x, point1_y, point3_x, point3_y)
    d2 = distanceBetweenTwoPoints(point1_x, point1_y, point4_x, point4_y)
    d3 = distanceBetweenTwoPoints(point2_x, point2_y, point3_x, point3_y)
    d4 = distanceBetweenTwoPoints(point2_x, point2_y, point4_x, point4_y)  
    d = min(d1, d2, d3, d4)
    #adaptive threshold
    spatial_proximity_threshold = l1 * spatial_distance_parameter    
    if(d <= spatial_proximity_threshold):
        #print d, spatial_proximity_threshold
        length_penalty = l2 / l1
        distance_penalty = d / spatial_proximity_threshold
        penalty = length_penalty + distance_penalty
        adaptive_angular_difference_threshold = (1 - (1 / (1 + math.exp(-2*(penalty-1.5))))) * angular_difference_threshold
        angle_between_lines = abs(angle(line_segment1) - angle(line_segment2))
        #print angle_between_lines, adaptive_angular_difference_threshold, angular_difference_threshold
        if( (angle_between_lines < adaptive_angular_difference_threshold) or (angle_between_lines > (180 - adaptive_angular_difference_threshold)) ):
            d5 = length(line_segment1)
            #finding end points (farthest distance)
            d = max(d1, d2, d3, d4, d5)
            if (d == d1):
                end_point1_x = point1_x
                end_point1_y = point1_y
                end_point2_x = point3_x
                end_point2_y = point3_y
            elif (d == d2):
                end_point1_x = point1_x
                end_point1_y = point1_y
                end_point2_x = point4_x
                end_point2_y = point4_y                
            elif (d == d3):
                end_point1_x = point2_x
                end_point1_y = point2_y
                end_point2_x = point3_x
                end_point2_y = point3_y 
            elif (d == d4):
                end_point1_x = point2_x
                end_point1_y = point2_y
                end_point2_x = point4_x
                end_point2_y = point4_y    
            else:
                end_point1_x = point1_x
                end_point1_y = point1_y
                end_point2_x = point2_x
                end_point2_y = point2_y    
            merged_line = [end_point1_x, end_point1_y, end_point2_x, end_point2_y]                          
            angle_merged_line = angle(merged_line)
            #print abs(angle(line_segment1) - angle(merged_line)), (0.5 * angular_difference_threshold)
            theta = abs(angle(line_segment1) - angle(merged_line))
            if ( (theta < (0.5 * angular_difference_threshold)) or (theta > (180 - (0.5 * angular_difference_threshold))) ):
                return merged_line
    return None
        
    
def mergeLines ( line_segments, spatial_distance_parameter, angular_difference_threshold ):
    i = 0
    while ( True ): 
        mergedFlag = 0       
        n = len(line_segments)      
        #sort line segments in descending order of length
        line_segments.sort(cmp = lambda x, y: 1 if length(x) < length(y) else -1)                        
        line_segment1 = line_segments[i]
        l1 = length(line_segment1)
        #adaptive threshold
        spatial_proximity_threshold = l1 * spatial_distance_parameter
        merging_set = set()
        for j in xrange(i + 1, len(line_segments)):
            line_segment2 = line_segments[j]
            if(abs(angle(line_segment2) - angle(line_segment1)) < angular_difference_threshold):
                merging_set.add(j)
        lines_to_remove = set()                
        for index in merging_set:
            line_segment2 = line_segments[index]
            horizontally_close = isHorizontallyClose(line_segment1, line_segment2, spatial_proximity_threshold)
            if(horizontally_close):
                vertically_close = isVerticallyClose(line_segment1, line_segment2, spatial_proximity_threshold)
                if(vertically_close): 
                    continue
            lines_to_remove.add(index)
        merging_set.difference(lines_to_remove)
        lines_to_remove = set()
        for index in merging_set:
            #print index
            line_segment2 = line_segments[index]
            merged_line = mergeTwoLines(line_segment1, line_segment2, spatial_distance_parameter, angular_difference_threshold)
            if (merged_line != None):
                line_segment1 = merged_line             
                lines_to_remove.add(index)
                mergedFlag = 1
        line_segments[i] = line_segment1
        lines_to_remove = sorted(lines_to_remove, reverse=True)
        for index in lines_to_remove:            
            #can optimize removal by specifying index
            del line_segments[index]
        if(not mergedFlag):             
            i = i + 1        
        #print i
        if(i >= len(line_segments)-1):
            break
    return line_segments
        

def plot ( image, lines ):
    for line in lines:
        x1 = int(float(line[0]))
        x2 = int(float(line[2]))
        y1 = int(float(line[1]))
        y2 = int(float(line[3]))
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)        
    cv2.imshow('image', image )
    #cv2.imwrite('10_after.jpg', image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    #with open('output_10.txt', 'r') as f:
        #line_segments = f.readlines()         
    #format of line segment:
    #x1, y1, x2, y2, width, p, -log(nfa)
    #remove trailing new line
    #line_segments = [x.strip().split(' ') for x in line_segments]            
    #remove blank line at end
    #line_segments = line_segments[:-1]
    
    #line_segments = [[250, 10, 250, 100],
                    #[240, 90, 240, 250]]
                    #[10, 125, 10, 140]]        
    #threshold parameters to tune
    #TODO: tune parameters
    im = b[9]
    img1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img1)[0]
    line_segments=[]
    for x in lines:
        line_segments.append(x[0])
    spatial_distance_parameter = 0.75
    angular_difference_threshold = 0.05
    
    
    print 'Original number of lines = ', len(line_segments)
    img = im
    img1 = im.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plot(img, line_segments)
    #cv2.imshow('10_before.jpg', img)
    #cv2.waitKey(0)
    merged_lines = mergeLines(line_segments, spatial_distance_parameter, angular_difference_threshold)
    print 'New number of lines = ', len(merged_lines)            
    with open('merged_lines.txt', 'w') as f:    
        for line in merged_lines:
            f.write(str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + " " + str(line[3]) + "\n")
    merged_lines.sort(cmp = lambda x, y: 1 if length(x) < length(y) else -1)            
    plot(img1, merged_lines[0:100])   
