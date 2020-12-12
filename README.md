# An Efficient Spiking Neural Network for Recognizing Gestures with a DVS Camera on the Loihi Neuromorphic Processor
This repository provides the source codes for analyzing the DNN-to-SNN conversion through SNNToolbox, and the codes for pre-processing the DvsGesture dataset to make it possible to train in the DNN domain. For more details, please refer to [our IJCNN '20 paper](https://ieeexplore.ieee.org/document/9207109). If you used these results in your research, please refer to the paper
```
R. Massa, A. Marchisio, M. Martina and M. Shafique, "An Efficient Spiking Neural Network for Recognizing Gestures with a DVS Camera on the Loihi Neuromorphic Processor," 2020 International Joint Conference on Neural Networks (IJCNN), Glasgow, United Kingdom, 2020, pp. 1-9, doi: 10.1109/IJCNN48605.2020.9207109.
```
```
@INPROCEEDINGS{Massa2020AnEfficientSNN,
  author={R. {Massa} and A. {Marchisio} and M. {Martina} and M. {Shafique}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={An Efficient Spiking Neural Network for Recognizing Gestures with a DVS Camera on the Loihi Neuromorphic Processor}, 
  year={2020},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/IJCNN48605.2020.9207109}}
```
## Using scripts
The folder `SNNToolbox_analysis` contains the scripts for analyzing the DNN-to-SNN conversion through SNNToolbox.
- `simulator.py` requires as input a DNN trained in Keras (in `.h5` format), and executes the conversion with the parameters specified in `sim_specs.txt`.
- `sweep_simulator.py` executes a sweep of the simulation with different parameters, specified in `sim_sweep_specs.txt`.
