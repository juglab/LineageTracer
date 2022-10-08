<p align="center">
  <img src="https://user-images.githubusercontent.com/34229641/194717401-89682898-a7a8-4c0c-8220-4b9cc43f7365.png" height=300 />
</p>
<h2 align="center">Deep Lineage Tracing in Microscopy</h2>

## Table of Contents

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Datasets](#datasets)**
- **[Training & Inference on your data](#training-and-inference-on-your-data)**
- **[Issues](#issues)**
- **[Acknowledgements](#acknowledgements)**


### Introduction
This repository hosts the version of the code used to perform tracking on 2D and 3D microscopy time-lapse acquisitions.
We refer to the techniques elaborated in the code, here as **LineageTracer**.

`LineageTracer` follows the tracking-by-detection paradigm where in the first step, cells or nuclei in a time-lapse movie are detected by an instance segmentation approach, and in the second step these segmentations are re-assigned labels to form a lineage tree.

For the second step, a Graph Neural Network is trained on a subset of training image frames where single-pixel tracking annotations are available and this trained model is later used during inference, for constructing a lineage tree from the test image frames of the time-lapse movie.


### Dependencies
We have tested this implementation using `pytorch` version 1.10.0 and `cudatoolkit` version 10.2 on a `linux` OS machine.
One could execute these lines of code to run this branch:

```
conda create -n LineageTracerEnv python==3.7
conda activate LineageTracerEnv
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/juglab/LineageTracer.git
cd LineageTracer
pip install -e .
```

### Getting Started


Look in the `examples` directory,  and try out the `Fluo-N2DH-GOWT1` notebooks for 2D images or `Fluo-C3DL-MDA231` notebooks for volumetric (3D) images. Please make sure to select `Kernel > Change kernel` to `LineageTracerEnv`.   

### Datasets

Datasets are available as release assets **[here](https://github.com/juglab/LineageTracer/releases/tag/v0.1.0)**. 

### Training and Inference on your data

`*.tif`-type images, corresponding segmentation masks and single pixel tracking annotations should be respectively present under `images`, `masks` and `tracking-annotations`, under directories `train` and `test`. The following would be a desired structure as to how data should be prepared.

```
$data_dir
└───$project-name
    |───train
        └───images
            └───t000.tif
            └───...
            └───t00n.tif
        └───masks
            └───mask000.tif
            └───...
            └───mask00n.tif
        └───tracking-annotations
            └───man_track000.tif
            └───...
            └───man_track00n.tif       
    |───test
        └───images
            └───...
        └───masks
            └───...
```

### Issues

If you encounter any problems, please **[file an issue]** along with a detailed description.

[file an issue]: https://github.com/juglab/LineageTracer/issues

### Acknowledgements

