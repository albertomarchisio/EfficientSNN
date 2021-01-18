import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B



gesture_path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture32x32_6ch.pickle'
(x_train,y_train),(x_test,y_test) = dvs_gesture_loader(gesture_path)

print(x_train.shape)
print(x_test.shape)

#for l in range(len(x_train)):
#    for x in range(32):
#        for y in range(32):
#            for c in range(6):
#                print(x_train[0][x,y,c])

"""
for l in range(len(x_test)):
    for x in range(32):
        for y in range(32):
            for c in range(6):
                if math.isnan(x_test[l,x,y,c]):
                    print("x_test nan at frame {} position ({},{}), channel {}".format(l,x,y,c))
"""
gesture_list = ["hand clapping",
                "R wave",
                "L wave",
                "R clockwise",
                "R counter clockwise",
                "L clockwise",
                "L counter clockwise",
                "arm roll",
                "air drums",
                "air guitar",
                "other gestures"]    

train_class_counter = np.zeros(11)
counter = 0
for y in y_train:
    for i in range(11):
        if y == i:
            train_class_counter[i] += 1
            counter += 1
test_class_counter = np.zeros(11)
for y in y_test:
    for i in range(11):
        if y == i:
            test_class_counter[i] += 1
            counter +=1

plt.subplot(2,1,1)
y_pos = np.arange(11)
plt.barh(y_pos, train_class_counter)
plt.yticks(y_pos,gesture_list, rotation = 0)
plt.title("train set")


plt.subplot(2,1,2)
y_pos = np.arange(11)
plt.barh(y_pos, test_class_counter, color = "g")
plt.yticks(y_pos,gesture_list, rotation = 0)
plt.title("test set")
plt.show()



"""
print(counter)


print("number of events per gesture:")
for i in range(11):
    print("{}: {}".format(gesture_list[i], (train_class_counter[i]+test_class_counter[i])/120))

print("number of events per seconds:")
for i in range(11):    
    print("{}: {}".format(gesture_list[i], (train_class_counter[i]+test_class_counter[i])/(120*4)))

"""