# A-Booth-Inspired-Approximation-on-Quantization-for-Neural-Networks

This repository is part of a paper submission.
Submitted for IEEE Transactions on Circuits and Systems for Artificial Intelligence.
Code provided under the LISCENCE attached to this repository.

## HDL
The folder *HDL* contains the Verilog definitions for the hardware presented in the paper.

*main_pe_area.tcl* is the script used to synthesize the hardware in Cadence Genus.

Usage:
- Replace the file list on line 5 (*read_hdl -sv {file.v}*) with either of the following
    * *{processing_element.v}*
    * *{processing_element_8x8.v}*
    * *{processing_element_16x1.v fp_add.v}*
    * *{processing_element_32x1.v fp_add.v}*
- Run *genus -f main_pe_area.tcl*

Reports from the execution of the above command, with all variants included, are within the corresponding report folder in the *HDL* folder.

## Simulation
The folder *py_simulation* contains the Python code for the simulations presented in the paper.

*quantize_toolkit.py* contains all the custom made code, the rest of the *.py* files are the main executables for each neural network.

Usage:
- Select your desired neural network:
    * *Alexnet.py*
    * *DenseNet.py*
    * *GoogLeNet.py*
    * *InceptionV3.py*
    * *MNIST_CNN.py*
    * *MNIST_MLP.py*
    * *Mobilenet.py*
    * *Resnet.py*
    * *VGG16.py*
    * or make your own following the structure of one of the files above
- Run it with the python command

The following functions in *quantize_toolkit.py* will save their results:
- apply_custom_quantization
- train
- PTQ_quantize
- QAT_quantize
And include a load boolean argument. Setting this argument to true, with previous results saved, will cancel any computations and load the existing results.