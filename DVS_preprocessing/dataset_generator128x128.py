import struct
import numpy as np
from numpy import unravel_index
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
np.set_printoptions(threshold=np.inf)

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


    return np.array(x_addr_tot), np.array(y_addr_tot), np.array(pol_tot), np.array(ts_tot), np.array(spec_type_tot), np.array(spec_ts_tot)

#--- frame_generator:
#   function used to generate all the frames from the collected events of a given hand gesture
def frame_generator(events_list, number_of_events, accumulation_set, numeric_label):
    # events_list structure:
    # - list of [x, y, p] for each event
    # x is in the interval [0, 128]
    # y is in the interval [0, 128]
    # p can be 0 (OFF) or 1 (ON)
    # ts_tot is not necessary for this task

    # number of frames given the defined accumulation_set
    n_frames = math.floor(number_of_events/accumulation_set)
    
    for i in range(n_frames):         
        # create two frame matrix: one for the positive events and one for negative events
        frame_ON  = np.zeros(shape=(128,128))  
        frame_OFF = np.zeros(shape=(128,128))
        #frame_PREDATOR = np.full((128, 128), 0.5)  # do what they did in the article "steering predator..."
        
        # assign polarity value to the two matrices
        
        if i != n_frames-1:
            subset_of_events = events_list[(i*accumulation_set) : ((i+1)*accumulation_set-1)]
        else: 
            subset_of_events = events_list[(i*accumulation_set) :]

        for event in subset_of_events:
            
            if event[2] == 1:                                                                                       # TRYING TO FIND GOOD ACCUMULATION 
                frame_ON[event[0],event[1]]  += 1   #sum event that happens in the same pixel
                #frame_PREDATOR[event[0],event[1]]  += 1/200   #sum event that happens in the same pixel
            else: 
                frame_OFF[event[0],event[1]] += 1   #sum event that happens in the same pixel
                #frame_PREDATOR[event[0],event[1]]  += -1/200   #sum event that happens in the same pixel


        # Normalization:
        max_ON  = np.amax(frame_ON)
        max_OFF = np.amax(frame_OFF)
        #max_PREDATOR = np.amax(frame_PREDATOR_resize)
        frame_ON= np.divide(frame_ON,max_ON)
        frame_OFF = np.divide(frame_OFF,max_OFF)  
        #frame_PREDATOR_resize = np.divide(frame_PREDATOR_resize,max_PREDATOR)  
        
        frame_ON = np.multiply(frame_ON,255)
        frame_OFF = np.multiply(frame_OFF,255)
        #frame_PREDATOR_resize = np.multiply(frame_PREDATOR_resize,255)

        frame = np.zeros(shape=(128,128,2))   # final 2-channel frame that will contain both the accumulation of off and on events

        for x in range(128):
            for y in range(128):
                for i in range(2):
                    if i == 0: 
                        frame[x,y,i] = frame_ON[x,y]     # ON events on channel 1
                    else:
                        frame[x,y,i] = frame_OFF[x,y]    # OFF events on channel 2
    
        # add the resulting frame and its corresponding label to the complete list of frames
        if train_set == True:
            x_train.append(frame)
            y_train.append(numeric_label)
        else: 
            x_test.append(frame)
            y_test.append(numeric_label)
#-- END frame_generator


# for DVS dataset
xdim = 128
ydim = 128
accumulation_set = 5000 #number of events that will produce a single frame
base_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/DvsGesture'
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
csv_mapping = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/DvsGesture/gesture_mapping.csv'

aedat_counter = 0   # to keep count of how many aedat file I have analyzed

start = time.time() # start time for measure conversion time

aedat_list = ["user01_led.aedat"]
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


    # reduce the duration of each each gesture to 1.5 seconds (as in the slayer paper)
    for key in key_list:
        start = int(start_stop_dict[key][0])                    # start time
        stop = int(start_stop_dict[key][1])                     # stop time
        duration = stop - start                                 # duration
        print("{} duration: {}".format(key, duration/1000000))

        #time_gap = duration-1500000                             # how much time I have to skip to get the center part of the video of the gesture
        #new_start = int(start + time_gap/2)                          # new start is half of the time_gap after the initial start --> I get that the part of the gesture that I take is centered
        #new_stop  = int(stop - time_gap/2)                           # new stop is half of the time_gap before the initial stop --> I get that the part of the gesture that I take is centered

        #print("old start: {} stop:{}".format(start, stop))
        #print("new start: {} stop:{}".format(new_start, new_stop))
        new_start = int(start + 500000)                          # new start is half of the time_gap after the initial start --> I get that the part of the gesture that I take is centered
        new_stop  = int(stop  - 500000)                           # new stop is half of the time_gap before the initial stop --> I get that the part of the gesture that I take is centered
        print("new duration: {}".format((new_stop-new_start)/1000000))
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

                for (x_, y_, p_) in zip(x,y,p):
                    events_info = [y_,x_,p_]        # store relevant infos, NB: switch x and y is needed because otherwise the image is 90 rotate
                    events_list.append(events_info) # create a list of all the events that will be used to make the frames
                
                number_of_events += len(ts_tot) # counts the total number of events of the hand gesture

                # check if the gesture events are finished:
                for ts in ts_tot:
                    if (abs(int(start_stop_dict[actual_gesture][1])<=int(ts))):    #stop collecting events of the gesture
                        collect_enabler = False
                        print("- {}: completed, {} frames collected".format(actual_gesture, math.floor(number_of_events/accumulation_set)))             
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
print("saving dataset as: dvs_gesture_128x128.pickle ...")
with open("dvs_gesture_128x128.pickle",'wb') as pickle_file:    
    pickle.dump(tot, pickle_file)
    pickle_file.close()
print("completed")