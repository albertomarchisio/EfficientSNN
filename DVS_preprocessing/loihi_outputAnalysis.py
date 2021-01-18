import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import matplotlib.animation as animation
import pickle
import cv2
import gc
#np.set_printoptions(threshold=np.nan)


def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B

net = "gestureNet1"
dataset = "gesture32x32"
duration = "256"

pickle_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_'+dataset+'.pickle'
results_path = ("/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/nxsdk_results/reports_accuracy/" + net + "/" + dataset + "/4572_images/nahuku32_2h/duration_" + duration + "/dThIR_1/ThNormOn/soft/output_spiketrains/")


# Import all the output spiketrains of all the neurons:
npy_list = list()
for file in os.listdir(results_path):
    if file.endswith(".npy"):
        npy_list.append(os.path.join(results_path, file))
npy_list.sort()


outputSpike_list = list() # list that will have as element an array that contains the output spikes history of the simulation, each element of the list correspond to the output spikes of a given frame 

for npy in npy_list:    # for each file I want to extract all the spike events and put them into a list
    
    spike = np.load(npy)            # each contains the spikes of the output layer for one frame, during a period of 256 timesteps
    outputSpike_list.append(spike)    


# import dataset labels
(_,_),(_,y_test) = dvs_gesture_loader(pickle_path)


# each element of the outputSpike_list contains the accumulation history of the output spikes for each frame during the simulation of a frame, where each simulation last 256 timestamp.
# Now I need to extend the history of the output spikes over the whole group of frames that are part of the same gesture.

frame_counter = 0 # counts the number of frames of a gesture
last_frame_used = 0 #   keep track of the total number of frames that have yet been collected in the final list of spikes history
total_frames = 0
complete_output_list = list()
true_label_list = list()        # list that contains the label of each gesture
for i in range(len(y_test)):
    if i == 0:
        frame_counter += 1
        continue
    else:
        if (y_test[i] != y_test[i-1]) or (i == len(y_test)-1):
            frame_counter += 1
            #compute the total number of timestamps elapsed during the gesture:
            # create the new matrix that will be used to record the history of the output spikes
            gesture_spikes = np.zeros((frame_counter,11)) 
            for c in range(frame_counter):      # iterate over the number of frames of a gesture
                for x in range(256):            # iterate over the 256 timesteps of a loihi prediction
                    for n in range(11):         # iterate over the 11 output neurons
                        spike = outputSpike_list[last_frame_used + c]       # select the output spike train of a given frame
                        if spike[0,n,x] == True:
                            gesture_spikes[c:,n] += 1       # increment the spike count of the gesture
            true_label_list.append(y_test[i-1])                                                                         
            complete_output_list.append(gesture_spikes)
            last_frame_used += frame_counter
            total_frames += frame_counter
            frame_counter = 0 

        else:
            frame_counter += 1  
        

print("total number of frames: {}".format(total_frames))

# CHECKPOINT: here I save the output list into a pickle file, such that the previous part of the code can be executed one time only
# save the output list of spike history in order to avoid to re-create it again another time (time consuming)
tot = list()
with open("output_spiketrains.pickle",'wb') as pickle_file:    
    tot.append(complete_output_list)
    tot.append(true_label_list)
    pickle.dump(tot, pickle_file)
    pickle_file.close()


complete_output_list = list()
true_label_list = list () 
with open("output_spiketrains.pickle",'rb') as pickle_file:     # import the previous generated output spikes history
        (complete_output_list, true_label_list) = pickle.load(pickle_file)

# END OF CHECkPOINT----------------------


# Compute the accuracy of the network:
accuracy=0
for output, label in zip(complete_output_list, true_label_list):
    if np.argmax(output[-1,:]) == label:
        accuracy += 1

accuracy = accuracy/len(true_label_list)*100


# compute the average number of frames needed to recognize the gesture:
avg_frames_to_rec = 0 
avg_number_of_frames = 0 
frames_to_rec     = 0 
correct_pred      = False
for output, label in zip(complete_output_list, true_label_list):
    avg_number_of_frames += output.shape[0]
    for i in range(output.shape[0]):
        predict = np.argmax(output[i,:])
        if predict != label:
            frames_to_rec = np.argmax(output[-1,:])        # reset
            if i == output.shape[0]-1:
                print("wrong\n")
            correct_pred = False
        
        elif predict == label:
            if correct_pred == False: # First time (or new time) in which I get the correct prediction
                if frames_to_rec == 0:
                    frames_to_rec = 1
                else:
                    frames_to_rec = i
                    if frames_to_rec > 4:
                        print("new max frame: {}".format(frames_to_rec))
                correct_pred = True
            else:
                continue
    avg_frames_to_rec += frames_to_rec
avg_frames_to_rec = avg_frames_to_rec/len(true_label_list)
avg_number_of_frames = avg_number_of_frames/len(true_label_list)
print("avg_frames_to_rec = {}".format(avg_frames_to_rec))
print("avg_number_of_frames = {}".format(avg_number_of_frames))
print(accuracy)


# animation function  for the bar plot of the spike accumulation
def animate(num,data,plot):
    max_height = 0
    for y in data[num,:]:
        if y > max_height:
            max_height = y

    for rect, y in zip(p1, data[num,:]):
        rect.set_width(y)
        # set the color of the higher column as red
        if y == max_height:
            rect.set_color('tab:red')
        else: 
            rect.set_color('tab:blue')
    return p1

gesture_list = ["hand clapping",
                "right hand wave",
                "left hand wave",
                "right arm clockwise",
                "right arm counter clockwise",
                "left arm clockwise",
                "left arm counter clockwise",
                "arm roll",
                "air drums",
                "air guitar",
                "other gestures"]    

counter = 0     # count the number of video created ( will be useful to use these video alongside the dvs gesture video)
for (output_spikes, true_label) in zip(complete_output_list,true_label_list):      # output spikes of a gesture
    numb_of_timesteps = output_spikes.shape[0]      # number of frames (each of which is a collection of 256 timesteps)

    # create the animation for the bar graph of the output spikes
    fig1 = plt.figure(figsize=(5,5))
    y_pos = np.arange(11)                   # output neurons
    p2 = plt.barh(true_label, output_spikes[numb_of_timesteps-1,:], 1, color = 'lightgreen', label = "True Label")
    p1 = plt.barh(y_pos,output_spikes[0,:])  # start the bar plot with the first spike accumulation of the first frame
    plt.xlabel("spike count",fontsize=15)
    plt.yticks(y_pos,gesture_list, rotation = 0)
    plt.tight_layout()

    legend_properties = {'weight':'bold'}
    leg = plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.02))
    leg.set_title("accuracy = {0:.2f}%".format(accuracy), prop = legend_properties)
    plt.xlim(0,np.amax(output_spikes[numb_of_timesteps-1,:])+100)

    line_ani = animation.FuncAnimation(fig1, animate, numb_of_timesteps, fargs=(output_spikes,p1), interval=100, repeat=False, blit=True)
    savePath = "/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/videoSteps/"
    
    line_ani.save((os.path.join(savePath,str(counter).zfill(6)+'.mp4')))
    counter += 1

# concatenate the video into one single video:
savePath = "/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/videoSteps/"
list_of_video = list()
for file in os.listdir(savePath):
    if file.endswith(".mp4"):
        list_of_video.append(os.path.join(savePath, file))
list_of_video.sort()
with open("list_of_video.txt","w") as f:
    for el in list_of_video:
        f.write("file '{}'\n".format(el))
    f.close()

command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'list_of_video.txt', '-c', 'copy', 'output_spike_video.mp4']
subprocess.call(command)
os.remove("list_of_video.txt")  # remove the txt file that is not useful anymore

# remove the single video and put the complete output_spike_video in the directory VideoSteps
for file in os.listdir(savePath):
    if file.endswith(".mp4"):
        os.remove(os.path.join(savePath, file))
#shutil.move("output_spike_video.mp4", os.path.join(savePath, "output_spike_video.mp4"))



#-------------------------------------------------------------------------
# merge the output_spike_video.mp4 with the test set video of the gestures
#-------------------------------------------------------------------------


# create the gesture video

#pickle_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture_32x32.pickle'
video_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture_test.avi'


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


(_,_),(x_set,y_set) = dvs_gesture_loader(pickle_path)
    

# create mp4 videos of gestures

# - 1) create all the png files of the single frames
# - 2) create the videos from the png frames

frame_list = list()         # list that will contain all the frames
frame_count = 0             # counter of the total number of frames

desired_shape = 500         # desired resolution of the output video
video_frame = np.zeros((32 ,32 ,3))   # video frames needs to have 3 channels, where the last one (the blue) is left at 0 

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
    out.write(frame_list[i])
    frame_list[i] = None

out.release()


# merge the two videos:

command = ["ffmpeg", "-i", "dvs_gesture_test.avi", "-i", "output_spike_video.mp4", "-filter_complex", "hstack", "loihi_"+ net+ dataset[-5:] + "_duration"+duration+".mp4"]

subprocess.call(command)
os.remove("dvs_gesture_test.avi")
os.remove("output_spike_video.mp4")

