import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


import sys
import keras
from keras.models import load_model

def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B

print("creating the inference video:")

pickle_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture32x32.pickle'
video_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture_inference.avi'


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

(_,_),(x_test,y_test) = dvs_gesture_loader(pickle_path)
    
# load the NN
net = "nxtfnet"
dataset = "gesture32x32"
duration = "256"
model_path = ("/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/nxsdk_simulations/trained_models/" + net + "_" + dataset + ".h5")
model = load_model(model_path)




# create mp4 videos of gestures

# - 1) create all the png files of the single frames
# - 2) create the videos from the png frames

frame_list = list()         # list that will contain all the frames
frame_count = 0             # counter of the total number of frames

desired_shape = 260         # desired resolution of the output video
video_frame = np.zeros((32 ,32 ,3))   # video frames needs to have 3 channels, where the last one (the blue) is left at 0 

for frame in x_test:
    # use directly the matrix to create the frames
    
    # assign the 2-channels frame to the 3-channels video frame
    for x in range(32):
        for y in range(32):
            video_frame[x, y, 1] = frame[x,y,0]   
            video_frame[x, y, 2] = frame[x,y,1]

    video_frame = video_frame.astype(np.uint8) # convert matrix to openCV compatible format

    #--- store the matrix frames for the video creation
    frame_list.append(video_frame)
    frame_count +=1

# modify the x_test and y_test to be used by the NN
x_test = x_test / 255.


# create the .avi
frame_rate=10

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'),frame_rate, (desired_shape,desired_shape))

print("creating video..")

preds = list()    # this is a list that will contain all the prediction in order to compute the rolling average and avoid the prediction flickering
reset_counter = 0 # counter used to reset the prediction list every once in a while such that the prediction on video is averaged over a small number of frames 
for i in range(len(frame_list)):
    # resize the frame to get to the desired output resolution
    frame_list[i] = cv2.resize(frame_list[i],(desired_shape, desired_shape))
    
    
    # evaluate the NN on the present frame:
    frame = x_test[i]       
    frame = np.expand_dims(frame, axis = 0)     # need to expand the dimension on the first axis because we are not working with batches 
    prediction = model.predict(frame)[0]

    preds.append(prediction)            # add the prediction to the list of all passed predictions
    results  = np.array(preds).sum(axis=0)  # average over the predictions to find the one that is more likely to be the correct prediction
    #results = np.array(preds).mean(axis=0)  # average over the predictions to find the one that is more likely to be the correct prediction

    predict_label = gesture_list[np.argmax(results)]

    reset_counter += 1          # increment the reset_counter
    #if reset_counter == 5:      # if 5 frames have been stacked, I reset the preds list
    if y_test[i] != y_test[i-1]: 
    #if reset_counter == 10:
        reset_counter = 0
        preds = list()

    # write the predicted label of the actual frame:
    text = predict_label
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (0, 44) 
    # fontScale 
    fontScale = 0.6
    # red color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 1 px 
    thickness = 1
    cv2.putText(frame_list[i],text,org,font,fontScale, color, thickness, cv2.LINE_AA) 

    # write real label of the actual frame:
    text = gesture_list[y_test[i]]
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
    cv2.putText(frame_list[i],text,org,font,fontScale, color, thickness, cv2.LINE_AA) 

    out.write(frame_list[i])
    frame_list[i] = None
out.release()