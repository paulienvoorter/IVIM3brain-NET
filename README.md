# IVIM3brain-NET

This repository contains the code regarding our submitted publication: Physics-informed neural networks improve three-component IVIM fitting in cerebrovascular disease

## Description
This code synthesizes three-component IVIM decay curves. Several fitting algorithms are implemented to fit the IVIM model to these synthesized curves: least squares (LSQ), non-negative least squares (NNLS) and physics-informed neural networks (PI-NN). 
Several options for the PI-NN architecture and IVIM parameter constraints are possible.

## Getting Started
To directly run the code, we added a '.yml' file which can be run in anaconda. To create a conda environment with the '.yml' file enter the command below in the terminal: conda env create -f environment.yml 

This now creates an environment called 'ivim' that can be activated by: conda activate ivim


Run Example_simulations.py to perform the simulations as described in our publication

### Dependencies
joblib=0.17.0
matplotlib=3.3.2
nibabel=3.2.0
numpy=1.19.2
python=3.6.12
pytorch=1.4.0
scipy=1.5
tqdm=4.50.2

## Authors
Paulien Voorter paulien.voorter@gmail.com | p.voorter@maastrichtuniversity.nl | https://github.com/paulienvoorter


## Version History
January 2022     Paulien Voorter
June 2021        Oliver Gurney-Champion and Misha Kaandorp https://github.com/oliverchampion/IVIMNET
August 2019      Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim

