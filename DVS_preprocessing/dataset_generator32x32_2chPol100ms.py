import struct
import numpy as np
import time
from time import sleep
from matplotlib import pyplot as plt
from matplotlib import style
import csv
import glob
import os
import math
import pickle
import time
import datetime

train_set = True    # flag that is used to define if the current file is a part of the train set

# the following list contain the frame and the labels of train and test set
x_train = list() 
y_train = list()
x_test = list() 
y_test = list()


def csv_open (csv_gesture, csv_mapping):
    # this function is used to open the csv files and takes the data.
    # returns a list of lists: each inner list contains:
    # ['*gesture_name*', 'start_time(usec)', 'stop_time(usec)']
    
    start_stop_list= list() # list that contains start and stop time of each gesture
    key_list = list() # list that contains the name of each gesture; these names will become the keys of the dictionary "start_stop_dict"

    with open(csv_gesture) as csv_file:
        csv_reader  = csv.reader(csv_file, delimiter = ',')
        line_count  = 0

        previous = list()   # the gesture 8 (arm rolling) is repeated two times in the dataset, so we need to remove the repetition and sum the two time sequence into a single interval 
        for row in csv_reader:
            if line_count == 0: # skip the first line of the file
                line_count += 1
                continue
            temp = list (row[1:])       #take start and stop times
            
            if (row[0] == previous):    # check if the previous gesture is equal to the present gesture (happens with arm rolling)
                start_stop_list[-1][1] = row[2]     # create a single time interval for the repeated gesture
            else:                       # in this case there is not a repetition, so simply add the time interval to the start_stop list
                start_stop_list.append(temp)
            previous = row[0]           # update the previous gesture with the present one
        
    with open(csv_mapping) as csv_file:    
        csv_reader  = csv.reader(csv_file, delimiter = ',')
        line_count  = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            temp = row[0]
            key_list.append(temp)

    start_stop_dict = {key: None for key in key_list}   # dictionary that contains, for each gesture, its start and stop times

    for (key, start_stop) in zip(key_list,start_stop_list):
        start_stop_dict[key] = start_stop
    return start_stop_dict, key_list


def skip_header(file_read):
    ''' skip header '''
    line = file_read.readline()
    while line.startswith(b'#'):
        if ( line == b'#!END-HEADER\r\n'):
            break
        else:
            line = file_read.readline()


def read_events(file_read):
    """ A simple function that read events from cAER tcp"""
    
    #raise Exception
    data = file_read.read(28)

    if(len(data) == 0 ):
        return [-1], [-1], [-1], [-1], [-1], [-1]

    # read header

    # struct.unpack() returns a tuple. I take only the first element of the tuple (by putting the [0] after the unpack)
    eventtype = struct.unpack('H', data[0:2])[0]    # 'H' stands for unsigned short, because is 2 Byte
    eventsource = struct.unpack('H', data[2:4])[0]
    eventsize = struct.unpack('I', data[4:8])[0]    # 'I' stands for unsigned short, because is 4 Byte
    eventoffset = struct.unpack('I', data[8:12])[0]
    eventtsoverflow = struct.unpack('I', data[12:16])[0]
    eventcapacity = struct.unpack('I', data[16:20])[0]
    eventnumber = struct.unpack('I', data[20:24])[0]
    eventvalid = struct.unpack('I', data[24:28])[0]
    next_read = eventcapacity * eventsize  # we now read the full packet
    data = file_read.read(next_read)    # I have moved to the [events] block, after the [header]
    counter = 0  # eventnumber[0]
    #return arrays
    x_addr_tot = []
    y_addr_tot = []
    pol_tot = []
    ts_tot =[]
    spec_type_tot =[]
    spec_ts_tot = []

    # eventType = 1 is a polarity event, so is ok!
    if(eventtype == 1):  # something is wrong as we set in the cAER to send only polarity events
        while(data[counter:counter + eventsize]):  # loop over all event packets
            aer_data = struct.unpack('I', data[counter:counter + 4])[0]
            timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0]
            x_addr = (aer_data >> 17) & 0x00007FFF
            y_addr = (aer_data >> 2) & 0x00007FFF
            x_addr_tot.append(x_addr)
            y_addr_tot.append(y_addr)
            pol = (aer_data >> 1) & 0x00000001
            pol_tot.append(pol)
            ts_tot.append(timestamp)
            #print (timestamp, x_addr, y_addr, pol)
            counter = counter + eventsize
    elif(eventtype == 0):   # eventType 0 is a special event
        spec_type_tot =[]
        spec_ts_tot = []
        while(data[counter:counter + eventsize]):  # loop over all event packets
            special_data = struct.unpack('I', data[counter:counter + 4])[0]
            timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0]
            spec_type = (special_data >> 1) & 0x0000007F
            spec_type_tot.append(spec_type)
            spec_ts_tot.append(timestamp)
            if(spec_type == 6 or spec_type == 7 or spec_type == 9 or spec_type == 10):
                print (timestamp, spec_type)
            counter = counter + eventsize
    elif(eventtype == 2):
        print("Frame Event")

    return np.array(x_addr_tot), np.array(y_addr_tot), np.array(pol_tot), np.array(ts_tot), np.array(spec_type_tot), np.array(spec_ts_tot)

#--- frame_generator:
#   function used to generate all the frames from the collected events of a given hand gesture
def frame_generator(events_list, number_of_events, accumulation_set, numeric_label):
    # events_list structure:
    # - list of [x, y, p, ts] for each event
    # x is in the interval [0, 128]
    # y is in the interval [0, 128]
    # ts_tot is not necessary for this task

    # EVENTS ACCUMULATION POLICY:--------------------
    # the final tensor will have shape 32x32x6:
    # Each of the 6 ouput channels contains 10ms accumulated events.
    # The final tensor will stack the 6 channels and form a single output frame. 

    # compute the number of frames:
    frame_duration = 100000          #100ms
    start    = events_list[0][3]
    end      = events_list[-1][3]
    lenght   = end-start
    n_frames = math.floor(lenght/frame_duration) # the number of frames is given by the division of the gesture duration (4seconds) by 60ms. That should be 66 frames
    # find the accumulation groups: 
    # for each frame, I have to accumulate a certain number of events, 
    # that depens on the number of events that are withing 60ms. At the same time, each channel will
    # contain a number of events equal to the events that happens within 10 ms.
    # Therefore, I need to find out which are the intervals of accumulation for each channel of each frame of each gesture.
    n_events_per_frame_tot = list() # list that will contain all the indecies of the events that will compose a single frame
    
    for i in range(len(events_list)):
        ts = events_list[i][3]

        # collect the number of events to be accumulated in each frame:
        if (ts-start) >= frame_duration:
            n_events_per_frame_tot.append(n_events_per_frame)
            start = ts  # update the start value such that the next frame will start counting from the following 60 ms
        else:
            n_events_per_frame = i
    
    inizio = time.time()
    for fr_count in range(n_frames):
        inizio = time.time()

        if fr_count == 0:   # first frame:
            subset_of_events = events_list[0 : n_events_per_frame_tot[fr_count]-1]
        elif fr_count == n_frames-1:  # last frame
            subset_of_events = events_list[n_events_per_frame_tot[fr_count-1] : ]
        else:   # intermidiate frames
            subset_of_events = events_list[n_events_per_frame_tot[fr_count-1] : n_events_per_frame_tot[fr_count]-1]

        frame = np.zeros((32,32,2))
       
        #create the channel matrix
        channel_ON   = np.zeros((128,128))  
        channel_OFF  = np.zeros((128,128))  

        for event in subset_of_events:
            if event[2] == 1:     
                    channel_ON[event[0],event[1]]  += 1   #sum event that happens in the same pixel
            else:     
                    channel_OFF[event[0],event[1]]  += 1   #sum event that happens in the same pixel
            

        # dimensionality reduction of the 2 matrices
        # I want to resize the image to 32x32, so I need to create 1 matrix with dimension 32x32, following the rule:

        resized_channel_ON  = np.zeros((32,32))      
        resized_channel_OFF  = np.zeros((32,32))      

        for i in range(0,128,4):
            for j in range(0,128,4):
                p_x = int(i/4)  # pixel x
                p_y = int(j/4)  # pixel y
                ONpolarity_count  = 0
                OFFpolarity_count  = 0
                for k in range(4):
                    for h in range(4):
                        ONpolarity_count  += channel_ON[i+k,j+h]
                        OFFpolarity_count  += channel_OFF[i+k,j+h]
                
                resized_channel_ON[p_x,p_y] =  ONpolarity_count/16 
                resized_channel_OFF[p_x,p_y] =  OFFpolarity_count/16 

        # Normalization:
        max_ON  = np.amax(resized_channel_ON)
        max_OFF  = np.amax(resized_channel_OFF)
        if max_ON!=0:
            resized_channel_ON = np.divide(resized_channel_ON,max_ON)
        if max_OFF!=0:
            resized_channel_OFF = np.divide(resized_channel_OFF,max_OFF)
        

        resized_channel_ON = np.multiply(resized_channel_ON,255)
        resized_channel_OFF = np.multiply(resized_channel_OFF,255)

        # add the channel to the output frame:
        for x in range(32):
            for y in range(32):
                frame[x,y,0] = resized_channel_ON[x,y]
                frame[x,y,1] = resized_channel_OFF[x,y]

        # add the resulting frame and its corresponding label to the complete list of frames
        if train_set == True:
            x_train.append(frame)
            y_train.append(numeric_label)
        else: 
            x_test.append(frame)
            y_test.append(numeric_label)

        fine = time.time()

        #print(datetime.timedelta(seconds=fine-inizio))
    print("gesture {} completed, {} frames".format(numeric_label, n_frames))


#-- END frame_generator


# for DVS dataset
xdim = 128
ydim = 128
accumulation_set = 6000 #number of events that will produce 6 channels of the output frame


import platform
print(platform.release())
if platform.release() == "5.3.0-26-generic":        # I am on my pc 
    base_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/DvsGesture'
else:       # I am on LOPO1
    base_path = '/srv/data/rmassa/from_my_pc/DVS_Gesture_dataset/DvsGesture'

#--- find all the aedat and csv file of the dataset:
aedat_list = list()
key_list = list() # list that contains the name of each gesture; these names will become the keys of the dictionary "start_stop_dict"
for file in os.listdir(base_path):
    if file.endswith(".aedat"):
        if file == 'user02_lab.aedat' or file == 'user12_fluorescent_led.aedat':      # check if the file is one of the damaged (from errata.txt)
            continue    # broken file will not be used
        else:
            aedat_list.append(file)

aedat_list.sort()

# open gesture mapping csv

import platform
print(platform.release())
if platform.release() == "5.3.0-26-generic":        # I am on my pc 
    csv_mapping = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/DvsGesture/gesture_mapping.csv'
else:       # I am on LOPO1
    csv_mapping = '/srv/data/rmassa/from_my_pc/DVS_Gesture_dataset/DvsGesture/gesture_mapping.csv'


aedat_counter = 0   # to keep count of how many aedat file I have analyzed

start = time.time() # start time for measure conversion time

for aedat in aedat_list:
    # check if the current file is for train or test: (test from user24 up)
    if aedat[4:6] == "24":
        train_set = False

    aedat_counter += 1
    #--- prepare the necessary variables and dictionary for the collection of events:
    class_counter    = 0             # this variable is used to count the classes as they are recognized during the extraction from the aedat file
    collect_enabler  = True          # flag that is set to True when the hand gesture is finished 
    start_collect    = False         # define if it is time to start collecting the events of a given gesture: needed to get gesture of 1.5s lenght
    end_of_file      = False         # flag that is set to True when the file is completly scanned
    gesture_counter  = 0             # counter that counts which gestures have been extracted yet 
    number_of_events = 0             # counts the total number of events of the hand gesture
    events_list      = list()        # list of all the collected events
    
    #--- open file:
     
    # open aedat file
    print('###############################################')
    print("FILE {}/{}: {}".format(aedat_counter,len(aedat_list),aedat))
    print('-----------------------------------------------')
    elapsed = time.time()-start
    elapsed_hms = str(datetime.timedelta(seconds=elapsed))
    print('|    time elapsed             = ' + elapsed_hms)
    estimated_remaining_time = (len(aedat_list)-aedat_counter)*(elapsed/aedat_counter)
    estimated_remaining_time_hms =str(datetime.timedelta(seconds=estimated_remaining_time))
    print('|    estimated remaining time = ' + estimated_remaining_time_hms)
    print('-----------------------------------------------')

    aedat = os.path.join(base_path,aedat)
    file_read = open(aedat, "rb") 
    skip_header(file_read)
    
    #open csv file
    csv_gesture = os.path.join(base_path, aedat[:-6]+'_labels.csv')

    # collect info from the csv files
    start_stop_dict, key_list = csv_open(csv_gesture, csv_mapping)  


    # reduce the duration of each each to a maximum duration of 4 seconds: this is done also in order to have a balanced dataset, and keep its size small

    for key in key_list:
        start = int(start_stop_dict[key][0])                    # start time
        stop = int(start_stop_dict[key][1])                     # stop time
        duration = stop - start                                 # duration
        #print("{} duration: {}".format(key, duration/1000000))

        if duration >= 4000001:
            time_gap = duration-4000000                             # how much time I have to skip to get the center part of the video of the gesture
            new_start = int(start + time_gap/2)                          # new start is half of the time_gap after the initial start --> I get that the part of the gesture that I take is centered
            new_stop  = int(stop - time_gap/2)                           # new stop is half of the time_gap before the initial stop --> I get that the part of the gesture that I take is centered

            #print("old start: {} stop:{}".format(start, stop))
            #print("new start: {} stop:{}".format(new_start, new_stop))
            #new_start = int(start + 500000)                          # new start is half of the time_gap after the initial start --> I get that the part of the gesture that I take is centered
            #new_stop  = int(stop  - 500000)                           # new stop is half of the time_gap before the initial stop --> I get that the part of the gesture that I take is centered
            #print("new duration: {}".format((new_stop-new_start)/1000000))
            #print("\n")
            start_stop_dict[key][0] = str(new_start)                # set the new start
            start_stop_dict[key][1] = str(new_stop)                 # set the new stop
        

    

    actual_gesture  = key_list[class_counter] # the actual gesture to be recognized

    while(end_of_file == False):
        if collect_enabler == True:
            x, y, p, ts_tot, spec_type, spec_type_ts = read_events(file_read)
            for ts in ts_tot: 
                if (abs(int(start_stop_dict[actual_gesture][0])>int(ts))):
                    start_collect = False 
                else: 
                    start_collect = True
                    break

            if start_collect == True:

                for (ts_, x_, y_, p_) in zip(ts_tot, x,y,p):
                    events_info = [y_,x_,p_, ts_]        # store relevant infos, NB: switch x and y is needed because otherwise the image is 90 rotate
                    events_list.append(events_info) # create a list of all the events that will be used to make the frames
                
                number_of_events += len(ts_tot) # counts the total number of events of the hand gesture

                # check if the gesture events are finished:
                for ts in ts_tot:
                    if (abs(int(start_stop_dict[actual_gesture][1])<=int(ts))):    #stop collecting events of the gesture
                        collect_enabler = False          
                        break

        if collect_enabler == False:

            
            # create the frames:
            frame_generator(events_list, number_of_events, accumulation_set, class_counter)
            
            # move to the next hand gesture
            number_of_events = 0
            class_counter  += 1
            if actual_gesture != "other_gestures":
                actual_gesture  = key_list[class_counter]
            else: 
                end_of_file = True
            events_list     = list()
            collect_enabler = True 
            start_collect   = False

    
# Final step: create the dataset from all the accumulated frames    
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


A = list()
B = list()
tot = list()

A.append(x_train)
x_train = None    # empty to save memory
A.append(y_train)
y_train = None    # empty to save memory
B.append(x_test)
x_test = None    # empty to save memory
B.append(y_test)
y_test = None    # empty to save memory


tot.append(A)
A = None    # empty to save memory
tot.append(B)
B = None    # empty to save memory

print('###############################################')
print("saving dataset as: dvs_gesture32x32_2chPol100ms.pickle ...")
with open("dvs_gesture32x32_2chPol100ms.pickle",'wb') as pickle_file:    
    pickle.dump(tot, pickle_file)
    pickle_file.close()
print("completed")
