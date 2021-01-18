import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B

set_creation = 'train'      # set this variable to define which video you want to create: the training set video or the test set video

print("creating the {} set video:".format(set_creation))

pickle_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture128x128.pickle'
video_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture_'+ set_creation +'Window.avi'


gesture_list = ["hand_clapping",
                "right_hand_wave",
                "left_hand_wave",
                "right_arm_clockwise",
                "right_arm_counter_clockwise",
                "left_arm_clockwise",
                "left_arm_counter_clockwise",
                "arm_roll",
                "air_drums",
                "air_guitar",
                "other_gestures"]


if set_creation == 'train':
    (x_set,y_set),(_,_) = dvs_gesture_loader(pickle_path)
    
if set_creation == 'test':
    (_,_),(x_set,y_set) = dvs_gesture_loader(pickle_path)
 

# create mp4 videos of gestures

# - 1) create all the png files of the single frames
# - 2) create the videos from the png frames

frame_list = list()         # list that will contain all the frames
frame_count = 0             # counter of the total number of frames

desired_shape = 128         # desired resolution of the output video
video_frame = np.zeros((128 ,128 ,3))   # video frames needs to have 3 channels, where the last one (the blue) is left at 0 

for frame in x_set:
    # use directly the matrix to create the frames
    
    # assign the 2-channels frame to the 3-channels video frame
    for x in range(128):
        for y in range(128):
            video_frame[x, y, 0] = frame[x,y,0]*255   
            video_frame[x, y, 2] = frame[x,y,1]*255

    video_frame = video_frame.astype(np.uint8) # convert matrix to openCV compatible format

    #--- store the matrix frames for the video creation
    frame_list.append(video_frame)
    frame_count +=1

x_set = None  # not needed anymore

# create the .avi
frame_rate=10

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'),frame_rate, (desired_shape,desired_shape))


# Compute center of mass:
centroid_list = list()

for frame in frame_list:
    
    Mx = 0
    My = 0
    mass = 0
    for i in range(128):
        for j in range(128):
            #if frame[i,j,1]<50:
            #    continue
            #else:
            Mx += j*frame[i,j,0]
            My += i*frame[i,j,0]
            mass += frame[i,j,0]
    centroid = (int(float(Mx)/mass) , int(float(My)/mass))
    
    centroid_list.append(centroid)

new_frame_list = list()
for i in range(len(frame_list)):
    frame = frame_list[i]
    (x_center, y_center) = centroid_list[i]
    print(x_center, y_center)
    if x_center >=32:
        x_start  = x_center-32   
    else:
        x_start  = 0

    if y_center >=32:
        y_start  = y_center-32   
    else:
        y_start  = 0     
    # 2) check that the last point is before 127
    if (127-x_start) < 64:
        x_start = 63
    if (127-y_start) < 64:
        y_start = 63
    print(x_start, y_start)
    for i  in range(64):
        for j in range(64):
            if i == 0 or j == 0 or i==63 or j==63:
                frame[y_start+i, x_start+j, 1] = 255
    new_frame_list.append(frame)
    
frame_list = new_frame_list       

print("creating video..")

for i in range(len(frame_list)):
    frame_list[i] = cv2.resize(frame_list[i],(desired_shape, desired_shape))
    #--- set a title of the video, changing for each gesture 
    
    text = "."
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org
    centroid = centroid_list[i]  
    org = (centroid[0], centroid[1]) 
    # fontScale 
    fontScale = 1
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 1 px 
    thickness = 1

    #cv2.putText(frame_list[i],text,org,font,fontScale, color, thickness, cv2.LINE_AA) 
    
    out.write(frame_list[i])
    frame_list[i] = None

out.release()