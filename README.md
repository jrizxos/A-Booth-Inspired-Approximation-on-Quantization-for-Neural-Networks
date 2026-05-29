# A-Booth-Inspired-Approximation-on-Quantization-for-Neural-Networks

[![DOI](https://zenodo.org/badge/1122199864.svg)](https://doi.org/10.5281/zenodo.20444104)

This repository is part of a paper submission.
Submitted for IEEE Access.
Code provided under the LISCENCE attached to this repository.

## HDL
The folder *HDL* contains the Verilog definitions for the hardware presented in the paper.

*main_dc.tcl* is the script used to synthesize the hardware in Synopsys Design Compiler.

Usage:
- Run *dc_shell -f main_dc.tcl*

Reports from the execution of the above command, are within the reports *folder* under the *HDL* folder.

## Simulation
The folder *py_simulation* contains the Python code for the simulations presented in the paper.

*quantize_toolkit.py* contains all the custom made code for quantizing and applying the approximation. *evaluate_toolkit.py* contains all the custom made code for evaluating the various neural networks. The rest of the *.py* files are the main executables for each neural network.

Usage:
- Install requirements:
    * pip install -r requirements.txt
- Choose your desired neural network model:
    * Alexnet.py 
    * AST.py 
    * BERT-large.py 
    * BLOOM-7b1.py 
    * DenseNet.py 
    * DistilBERT.py 
    * GoogLeNet.py 
    * GPT2-xl.py 
    * InceptionV3.py 
    * mms-large.py 
    * MNASNet.py 
    * MNIST_CNN.py 
    * MNIST_MLP.py 
    * Mobilenet.py 
    * OPT-13B.py 
    * Resnet.py 
    * SqueezeNet.py 
    * VGG16.py 
    * ViT.py 
    * Whisper-large.py
    * or make your own following the structure of one of the files above
- Run it with the python command:
    * python \<model\>.py

The following functions in *quantize_toolkit.py* will save their results:
- apply_custom_quantization
- train
- PTQ_quantize
- QAT_quantize
And include a load boolean argument. Setting this argument to true, with previous results saved, will cancel any computations and load the existing results.

## Full results
Due to space constraints in the paper, a summarized version of the results are published. The full results data are present in the *Full Results.xlsx* excel file in the root of this repository.
