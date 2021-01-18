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


def resize(_set, size):
    new_set = list()
    for frame in _set:
        resized = np.resize(frame,(size,size,2))
        new_set.append(resized) 
    new_set = np.asarray(new_set)
    return new_set


set_creation = 'test'      # set this variable to define which video you want to create: the training set video or the test set video

print("creating the {} set video:".format(set_creation))

pickle_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture32x32.pickle'
video_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture_'+ set_creation +'_resized36.avi'


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
 
print(x_set[0].shape)

 
#x_set = x_set[:1000]
x_set = resize(x_set[:1000],36)



# create mp4 videos of gestures

# - 1) create all the png files of the single frames
# - 2) create the videos from the png frames

frame_list = list()         # list that will contain all the frames
frame_count = 0             # counter of the total number of frames

desired_shape = 256         # desired resolution of the output video

video_frame = np.zeros((36 ,36 ,3))   # video frames needs to have 3 channels, where the last one (the blue) is left at 0 

for frame in x_set:
    # use directly the matrix to create the frames
    
    # assign the 2-channels frame to the 3-channels video frame
    for x in range(32):
        for y in range(32):
            video_frame[x, y, 0] = frame[x,y,0]
            video_frame[x, y, 2] = frame[x,y,1]

    video_frame = video_frame.astype(np.uint8) # convert matrix to openCV compatible format

    #--- store the matrix frames for the video creation
    frame_list.append(video_frame)
    frame_count +=1

x_set = None  # not needed anymore

# create the .avi
frame_rate=10

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'),frame_rate, (desired_shape,desired_shape))

print("creating video..")

for i in range(len(frame_list)):
    frame_list[i] = cv2.resize(frame_list[i],(desired_shape, desired_shape))
    #--- set a title of the video, changing for each gesture 
    
    text = gesture_list[y_set[i]]
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (0, 22) 
    # fontScale 
    fontScale = 0.6
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 1 px 
    thickness = 1

    #cv2.putText(frame_list[i],text,org,font,fontScale, color, thickness, cv2.LINE_AA) 

    out.write(frame_list[i])
    frame_list[i] = None
out.release()