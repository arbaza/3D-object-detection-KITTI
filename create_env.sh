#!/bin/bash

conda activate base
current=$(conda env list | grep "CMPT726")
if [[ -n $current ]]; then
  conda remove --name CMPT726 --all
fi
conda create --name CMPT726
conda activate CMPT726

sudo apt-get update
conda install python=3.7.2
conda install opencv
conda install -c conda-forge pybind11
conda install pyyaml
conda install numpy pandas matplotlib
conda install scipy pillow
conda install -c anaconda pil
conda install -c pytorch pytorch=1.0
conda install -c pytorch torchvision
conda install typing
conda install setuptools
sudo apt-get install libboost-all-dev
sudo pip install mayavi
sudo pip install PyQt5
sudo pip install gnuplot

