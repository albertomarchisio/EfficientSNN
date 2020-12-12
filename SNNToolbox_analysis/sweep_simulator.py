import os
import subprocess
import sys
import shutil

# This script is launched in case I want to execute a sweep simulation over various parameters.
# To do so, I iteratively modify the sim_specs.txt file and launch the simulator.py script.

sim_sweep_specs = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'sim_sweep_specs.txt'))
sim_specs = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'sim_specs.txt'))

with open(sim_sweep_specs, 'r') as f:
    lines = f.read().splitlines()

    tf_models =      lines[0].split()
    datasets =       lines[1].split()
    images_to_test = lines[2].split()
    loihi_devices =  lines[3].split()
    power_sets =     lines[4].split()
    sim_durations =  lines[5].split()
    dThIRs  =        lines[6].split()
    enable_plotss  = lines[7].split()
    th_norms =       lines[8].split()
    reset_modes =     lines[9].split()
    sweep_simulation =                 lines[10].split()
    


command_list = ['python3','-m','my_simulations.snntoolbox.nxsdk_simulations.simulator']

#clean up the temp directories before starting the sweep
temp_power = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'results_power','temp'))
temp_plots = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'results_plots','temp'))
temp_accuracy = os.path.abspath(os.path.join('..','nxsdk-apps2', 'my_simulations', 'snntoolbox', 'nxsdk_simulations', 'results_accuracy','temp'))
temp_list = [temp_power,temp_plots,temp_accuracy]
for temp in temp_list:
    if os.path.isdir(temp):
        shutil.rmtree(temp)

sim_count = 1
total_num_of_sims = len(tf_models)*len(datasets)*len(images_to_test)*len(loihi_devices)*len(power_sets)*len(sim_durations)*len(dThIRs)*len(enable_plotss)*len(th_norms)*len(reset_modes)

for tf_model in tf_models:
    for dataset in datasets:
        for image_to_test in images_to_test:
            for loihi_device in loihi_devices:
                for power_set in power_sets:
                    for sim_duration in sim_durations:
                        for dThIR in dThIRs:
                            for enable_plots in enable_plotss:
                                for th_norm in th_norms:
                                    for reset_mode in reset_modes:
                                        with open(sim_specs, 'w') as f:
                                            f.write(tf_model      + '\n')
                                            f.write(dataset       + '\n')
                                            f.write(image_to_test + '\n')
                                            f.write(loihi_device  + '\n')
                                            f.write(power_set     + '\n')
                                            f.write(sim_duration  + '\n')
                                            f.write(dThIR         + '\n')
                                            f.write(enable_plots  + '\n')
                                            f.write(th_norm       + '\n')
                                            f.write(reset_mode + '\n')
                                            f.write(sweep_simulation[0])
                                        print("##################################################################################################################")    
                                        print("########################################### STARTING SIMULATION {} OF {} ###########################################".format(str(sim_count), str(total_num_of_sims)))
                                        print("##################################################################################################################")    
                                        sim_count +=1
                                        subprocess.call(command_list)
                                        


