import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B


path = '/home/riccardo/Documents/uni/python/thesis/nxsdk/dnn_models/snntoolbox/DVS_Gesture_dataset/dvs_gesture32x32.pickle'

(x_train,y_train),(x_test,y_test) = dvs_gesture_loader(path)


print(x_test[0].shape)
"""
for test in x_test:
    plt.figure()
    plt.matshow(test[:,:,0], cmap='gray')
    plt.title(test)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
"""


