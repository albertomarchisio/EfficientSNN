from __future__ import print_function
import os
import math
import subprocess
import sys
import pickle
import shutil
import configparser
import numpy as np
import keras
from keras.datasets import cifar10, mnist, fashion_mnist
from snntoolbox.bin.run import main
import numpy as np

# function needed to correctly takes images from the dvs_gesture dataset

def dvs_gesture_loader(path):
    A = list()
    B = list()
    
    with open(path,'rb') as pickle_file:  
        A,B = pickle.load(pickle_file)

    return A, B

def line_table_printer(string,desired_lenght):
    print(string, end = "")
    space_needed = desired_lenght-len(string)
    for _ in range(space_needed):
        print(" ",end="")
    print("|")
    for _ in range(desired_lenght):
        print("_",end="")
    print("")
    

#------------------------------------------------------------------------------------------------------------------------------------
# USER SPECS SECTION:----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# find the specs in the sim_specs.txt file
specs_list = list()
sim_specs = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'sim_specs.txt'))
with open(sim_specs, 'r') as f:
    lines = f.read().splitlines()        
    # user defines the parameters of the simulation passing directly when the script is launched
    tf_model = lines[0] # lenet5, kerastutorialcifar10net..
    specs_list.append(tf_model)

    dataset = lines[1]
    specs_list.append(dataset)
    
    image_to_test = int(lines[2]) # can be either 1, 200, 1000 ecc.. 
    specs_list.append(lines[2] + '_images')

    loihi_device = lines[3]
    specs_list.append(lines[3])

    power_set = lines[4] # can be or pow or nopow

    sim_duration = int(lines[5]) # can be 256 ecc..   
    specs_list.append('duration_' + lines[5]) 

    dThIR = 2**int(lines[6]) #can be 0,1...10
    specs_list.append('dThIR_' + lines[6])

    enable_plots = lines[7] #can be either:  plots or noplots
    
    th_norm = lines[8] #can be ThNormOn or ThNormOff
    specs_list.append(lines[8])

    reset_mode = lines[9]   #can be 'soft', 'hard' (reset by subtraction and reset to zero respectively)
    specs_list.append(lines[9])

    sweep_simulation = lines[10]    # can either be sweep or nosweep


tf_model_name = tf_model + '_' + dataset #need to concatenate the two strings to get the correct name of the tf model 

if power_set == 'pow':      # needed in the config file of SNN TB
    power_analysis = True
else:
    power_analysis = False

if th_norm == 'ThNormOn':
    th_norm = True
else: 
    th_norm = False


if enable_plots == 'plots':      # needed in the config file of SNN TB
    enable_plots = True
    image_to_test = 1
else:
    enable_plots = False

if tf_model == "capsnet0":
    capsnet = True
else: 
    capsnet = False
#####################################################################################################################################


#------------------------------------------------------------------------------------------------------------------------------------
# SAVE PATH SECTION:-----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#-- first case: I want to do a power analysis:
if power_analysis == True:
    savePath_0 = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'reports_power'))

#-- second case: I want to do a plot analysis (Pearsons coefficients, correlation plots..):
elif power_analysis==False and enable_plots == True:
    savePath_0 = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'reports_plots'))

#-- third case: I want to do an accuracy analysis:
else:
    savePath_0 = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'reports_accuracy'))

if not os.path.isdir(savePath_0):
    os.mkdir(savePath_0)

# create the branch of directories
savePath = savePath_0

for el in specs_list:                           # iterative creation of directories 
    savePath = os.path.join(savePath, el)
    if not os.path.isdir(savePath):
        os.mkdir(savePath)

# I want to be sure that the last dir ( that is the one that will contains all the simulation file) is brand new and does not contain file of a previous simulation
shutil.rmtree(savePath)
os.mkdir(savePath)

#####################################################################################################################################

#------------------------------------------------------------------------------------------------------------------------------------
# GET DATASET SECTION:---------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
num_classes   = 10

# The data, split between train and test sets:
if (dataset=='mnist'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif (dataset=='cifar10'):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif(dataset=='cifar10dvs'):
    cifar10_dvs_path = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox','cifar10_dvs','cifar10_dvs.pickle'))
    (x_train, y_train), (x_test, y_test) = dvs_gesture_loader(cifar10_dvs_path)

elif(dataset[:7]=='gesture'):
    dvs_gesture_path = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox','DVS_Gesture_dataset','dvs_'+ dataset + '.pickle'))
    print(dataset)
    (_,_), (x_test, y_test) = dvs_gesture_loader(dvs_gesture_path)
    num_classes = 11

elif (dataset=='fashion_mnist'):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



if capsnet == True: #mi sa che non serve
    print(x_train.shape)
    x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255.
    print(x_train.shape)
    x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255.


elif dataset[:7]=='gesture':
    print("gesture!")
    x_test = x_test / 255.
    
    # Take only first half of the dataset:
    #x_test = x_test[10258:]
    #y_test = y_test[10258:]
    
else:   
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

if dataset == 'mnist' or dataset == 'fashion_mnist' or dataset == 'cifar10dvs':
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
#####################################################################################################################################


#------------------------------------------------------------------------------------------------------------------------------------
# PREPARE SNN TB:-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(savePath, 'x_test'), x_test)
np.savez_compressed(os.path.join(savePath, 'y_test'), y_test)

# copy the .h5 model il the working directory
tfModelPath = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'trained_models', tf_model_name + '.h5')) #dir where is the pre-trained tf model
ModelSavePath = os.path.abspath(os.path.join(savePath, tf_model_name+'.h5'))
shutil.copy2(tfModelPath, ModelSavePath)

# Create a config file with exerimental setup for SNN Toolbox.
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': savePath,             # Path to model.
    'dataset_path': savePath,        # Path to dataset.
    'filename_ann': tf_model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'parse': True,                  # Parses input model
    'normalize': False              # Not needed for Loihi backend.
}

if sim_duration <256:
    num_to_test = image_to_test
else: 
    num_to_test = 1

config['simulation'] = {
    'simulator': 'loihi',           # Chooses execution backend of SNN toolbox.
    'duration': sim_duration,               # Number of time steps to run each sample.
    'num_to_test': num_to_test,              # How many samples to run.
    'batch_size': 1,                # Batch size 1 is the only supported value.
    'keras_backend': 'tensorflow'}


config['restrictions'] = {
    'spiking_layers': {'Dense', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D', 'DepthwiseConv2D'} # Currently supported layers
}
# RM--------------
# I add this configuration option to disable the power analysis, disabling PROBE_ENERGY and PROBE_EXECUTION_TIME
#config['power_sim'] = {
#    'power_analysis' : power_analysis}

config['sim_type'] = {    
    'capsnet': capsnet}
# RM-end-----------

config['loihi'] = {
    # Set partition to run on (optional). Others: 'wm', 'nahuku08', ...
    'partition': os.environ.get('PARTITION', loihi_device),
    'validate_partitions': False,  # Validates correctness of compiled layers
    'save_output': False,          # Plots compiler information
    'use_reset_snip': True,        # Using snip accelerates reset between samples
    'do_overflow_estimate': True,  # Estimate overflow of dendritic accumulator.
    'normalize_thresholds': th_norm,  # Tune thresholds to optimal dynamic range.
    'desired_threshold_to_input_ratio': dThIR,
    # For DNNs it is recommended to set the biasExp = 6 and weightExponent = 0.
    'compartment_kwargs': {'biasExp': 6, 'vThMant': 2 ** 9},
    'connection_kwargs': {'numWeightBits': 8, 'weightExponent': 0, 'numBiasBits': 12},
    'reset_mode' : reset_mode,      # reset_mode can be 'soft'/'hard' for reset-by-subtraction (more accurate) or reset-to-zero
    'num_cores_per_chip': 128,  # Limites the maximum number of usale cores per chip
    'saturation': 6,                               # Saturation level of ReLU function used in ANN
    'plot_histograms': True,               # Create histrams of weights/biases of converted SNN
}

if enable_plots == True :
    config['output'] = {
        'plot_vars': {                  # Various plots (turns on probes, costly!)
            #'spiketrains',
            #'spikerates',
            #'activations',
            'correlation',
            #'v_mem',
            #'error_t'
            }
    }


# Store config file.
config_filepath = os.path.join(savePath, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)
#####################################################################################################################################



#------------------------------------------------------------------------------------------------------------------------------------
# RUN SNN TOOLBOX:-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#create a small recap of the simulation with all parameters:
print("############################################################################################")
print("##################################### SIMULATION RECAP #####################################")
print("############################################################################################")
print("____________________________________________________________________________________________")
string = ("|    simulating:               {}".format(tf_model)) 
line_table_printer(string,93)
string = ("|    dataset:                  {}".format(dataset))  
line_table_printer(string,93)
string = ("|    number of images:         {}".format(str(image_to_test)))  
line_table_printer(string,93)
string = ("|    device:                   {}".format(loihi_device))
line_table_printer(string,93)
if power_analysis == True:
    string = ("|    Power Analysis:           ON")
else:
    string = ("|    Power Analysis:           OFF")
line_table_printer(string,93)
string = ("|    duration:                 {}".format(str(sim_duration)))  
line_table_printer(string,93)
string = ("|    dThIR:                    2**{}".format(str(int(math.log2(dThIR)))))
line_table_printer(string,93)
if enable_plots == True:
    string = ("|    Correlation Plots:        ON")
else:
    string = ("|    Correlation Plots:        OFF")
line_table_printer(string,93)
if th_norm == True:
    string = ("|    Threshold Normalization:  ON")
else:
    string = ("|    Threshold Normalization:  OFF")
line_table_printer(string,93)
string = ("|    reset mode:               {}".format(reset_mode))
line_table_printer(string,93)
if sweep_simulation == 'sweep':
    string = ("|    Sweep simulation")
else:
    string = ("|    single simulation")
line_table_printer(string,93)
print("############################################################################################\n")


main(config_filepath)

#####################################################################################################################################



#------------------------------------------------------------------------------------------------------------------------------------
# RUN NxSDK SIMULATION:-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
def build_snn_from_file(num_steps_per_img):        #RM- modified
    #Loads converted SNN from file and builds composable DNN with InputGenerator.

    #:param num_step_per_img: Number of time steps a single sample is presented.

    #:returns Model with InputGenerator and SNN
    

    from nxsdk_modules.dnn.src.dnn_layers import loadNxModel
    from nxsdk_modules.dnn.composable.composable_dnn import ComposableDNN as CDNN
    from nxsdk_modules.input_generator.input_generator import InputGenerator
    from nxsdk.composable.model import Model

    # Define directory from which to load compiled SNN
    abs_log_dir = os.path.join(savePath, 'log', 'gui', 'test')
    model_path = os.path.join(abs_log_dir, 'model_dumps', 'runnables', 'nxModel.h5')

    # Load SNN from file
    snn_model = loadNxModel(
        filepath=model_path,
        customObjects=None,
        logdir=abs_log_dir)

    # Wrap loaded SNN by ComposableDNN class that allows to connect the SNN to input generator
    # (Wrapper won't be necessary anymore in future)
    # ComposalbeDNN needs num_steps_per_image because the readout snip and activity reset snip are executed
    # periodically after each sample
    cdnn = CDNN(model=snn_model, num_steps_per_img=num_steps_per_img)

    # Configure input generator to stream images via channels from super host to Loihi
    input_generator = InputGenerator(shape=snn_model.layers[0].input_shape,
                                     interval=num_steps_per_img,
                                     numSnipsPerChip=3)

    cdnn.name = "dnn"
    input_generator.name = "input"

    input_generator.setBiasExp(6)

    # All all components to parent model and connect input generator to SNN.
    model = Model('dnn')
    model.add(cdnn)
    model.add(input_generator)
    input_generator.connect(cdnn)
    # Enforce ordering of input and reset snip. The reset must execute before new input is injected.
    input_generator.processes.inputEncoder.executeAfter(cdnn.processes.reset)

    model.compile()

    return model

def run_model(model, sim_duration, x_test, y_test):
    """Runs the SNN Model to classify test images."""

    import time

    num_samples = len(y_test)

    # Start DNN
    tStart = time.time()
    model.start(model.board)
    model.run(sim_duration * num_samples, aSync=True)
    tEndBoot = time.time()

    # Enqueue images by pushing them into InputGenerator
    print("Sending images...")
    labels = np.zeros(num_samples, int)
    for i, (x, y) in enumerate(zip((x_test * 255).astype(int), y_test)):
        if len(x.shape) < 4:
            x = np.expand_dims(x, 0)
        model.composables.input.encode(x)
        labels[i] = np.argmax(y)
    tEndInput = time.time()

    # Read out classification results for all images
    print("Waiting for classification to finish...")
    classifications = list(model.composables.dnn.readout_channel.read(num_samples))
    tEndClassification = time.time()

    # finishRun fetches EnergyProbe data if configured
    model.finishRun()

    dt_boot = tEndBoot - tStart
    dt_input = tEndInput - tEndBoot
    dt_classification = tEndClassification - tEndBoot

    return (dt_boot, dt_input, dt_classification), classifications, labels

def calc_accuracy(classifications, labels):
    """Calculates classification accuracy for a set of images given classification and labels."""

    errors = classifications != labels
    num_errors = np.sum(errors)
    num_samples = len(classifications)
    return (num_samples - num_errors) / num_samples

if sim_duration>=256:
    # Classify
    if enable_plots == False:
        
        os.environ["SLURM"] = "1"
        model = build_snn_from_file(num_steps_per_img=sim_duration)
                                
        os.environ["PARTITION"] = loihi_device
        #os.environ["BOARD"] = "ncl-ghrd-04"

        # Select test subset
        np.random.seed(0)
        idx = np.random.choice(len(y_test), image_to_test, replace=False)


        (dt_boot, dt_input, dt_classification), classifications, labels = run_model(
            model,
            sim_duration,
            x_test[idx, :, :],
            y_test[idx])
        accuracy_path = os.path.join(savePath, 'log','gui','test', "report_nxsdk.txt")    
        with open(accuracy_path, 'w') as f:
            f.write("Time to boot: {0:.3f} s\n".format(dt_boot))
            f.write("Time to send input: {0:.3f} s\n".format(dt_input))
            f.write("Time to classify: {0:.3f} s\n".format(dt_classification))

            f.write("Average time to send individual input: {0:.3f} ms\n".format(
                dt_input/image_to_test*1000))
            f.write("Average time per classification: {0:.3f} ms\n".format(
                dt_classification/image_to_test*1000))
            f.write("Average time per classification time step: {0:.3f} us\n".format(
                dt_classification/image_to_test/sim_duration*1e6))

            f.write("Samples per second: {0:.3f}\n".format(image_to_test/dt_classification))

            accuracy = calc_accuracy(classifications, labels)
            f.write("Classification accuracy on samples: {0:.3f} %\n".format(accuracy*100))

        model.disconnect()


#------------------------------------------------------------------------------------------------------------------------------------


"""
# Build confusion matrix

import matplotlib.pyplot as plt

confusion = np.zeros((10, 10), dtype=int)
for l, c in zip(labels, classifications):
    confusion[c, l] += 1

# Normalize columns
col_sum = np.sum(confusion, axis=0)
confusion = confusion / col_sum

# Show confusion matrix
plt.figure(figsize=(5, 5))
plt.imshow(confusion)
plt.colorbar()
plt.xlabel("Label")
plt.ylabel("Classification")
_ = plt.title("Confusion matrix")
"""


#------------------------------------------------------------------------------------------------------------------------------------
# COLLECT USEFUL RESULTS:------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
dir_test = os.path.join(savePath, 'log','gui','test')

shutil.rmtree(os.path.join(dir_test,'output_spiketrains'))


temp = os.path.join(savePath_0,'temp')

# if I am doing a single simulation, I want to remove the results of the previous simulations inside the temp directory.
if sweep_simulation == "nosweep":
    if os.path.isdir(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)

# if I am executing a sweep of simulations, I want to accumulate the results of the different simulations inside the temp directory.
else:
    if not os.path.isdir(temp): # check, if the temp directory has not been created yet, create it
        os.mkdir(temp)

for el in specs_list:                           # iterative creation of directories 
    temp = os.path.join(temp, el)
    if not os.path.isdir(temp):
        os.mkdir(temp)

# I want to be sure that the last dir ( that is the one that will contains all the simulation file) is brand new and does not contain file of a previous simulation        
shutil.rmtree(temp) #the same directory will be created again with copytree

if os.path.isdir(dir_test):  # extract results only if the simulation is endend correctly and the directory has been created

    shutil.copytree(dir_test, temp)
    shutil.rmtree(os.path.join(temp,'log_vars'))
    shutil.rmtree(os.path.join(temp,'model_dumps'))
    shutil.rmtree(os.path.join(temp,'normalization'))
    
    #shutil.make_archive(os.path.join(temp,'output_spiketrains'), 'zip', temp, "output_spiketrains")

#####################################################################################################################################

