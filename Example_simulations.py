#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component model fitting of intravoxel incoherent motion MR imaging in cerebrovascular disease (2022)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# import
import numpy as np
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp_example
import time
from matplotlib import pyplot as plt
import os

# load hyperparameters
arg = hp_example()
arg = deep.checkarg(arg)



pathresults = '{folder}/results'.format(folder=os.getcwd())
# Check whether the specified path exists or not
isExist = os.path.exists(pathresults)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(pathresults)

start_time = time.time()

print('\n Simulation at SNR of {snr}\n'.format(snr=arg.sim.SNR))
if arg.fit.do_fit:
    matlsq, matNN, matnnls, stability = sim.sim(arg.sim.SNR, arg)
else:
    matNN, stability = sim.sim(arg.sim.SNR, arg)

# if repeat is higher than 1, then print stability (stability was not explored in publication)
if arg.sim.repeats > 1:
    print('\nstability of NN for Dpar, fint, Dint, fmv and Dmv:')
    print(stability)
    
elapsed_time = time.time() - start_time
print('\nTotal time elapsed: {} minutes\n'.format(elapsed_time/60))

#save results
if arg.fit.do_fit:
    np.save('{}/results-PINN-lr{}_ensemble{}'.format(pathresults,arg.train_pars.lr, arg.sim.n_ensemble), matNN)
    np.save('{}/results-LSQ'.format(pathresults), matlsq)
    np.save('{}/results-NNLS'.format(pathresults), matnnls)
else:
    np.save('{}/results-PINN-lr{}_ensemble{}'.format(pathresults,arg.train_pars.lr, arg.sim.n_ensemble), matNN)
    
#plot correlation matrix of parameter dependencies (not explored in publication)
#PINN
rhomatrix = [[1,round(matNN[3][3],2),round(matNN[1][4],2),round(matNN[2][4],2),round(matNN[0][4],2)],
             [round(matNN[3][3],2),1,round(matNN[4][3],2),round(matNN[0][3],2),round(matNN[3][4],2)],
             [round(matNN[1][4],2),round(matNN[4][3],2),1,round(matNN[4][4],2),round(matNN[1][3],2)],
             [round(matNN[2][4],2),round(matNN[0][3],2),round(matNN[4][4],2),1,round(matNN[2][3],2)],
             [round(matNN[0][4],2),round(matNN[3][4],2),round(matNN[1][3],2),round(matNN[2][3],2),1]]  
fig, ax = plt.subplots()
params = ['Dpar', 'Dint', 'Dmv', 'fint','fmv']
intersection_matrix = np.array(rhomatrix)
ax.matshow(intersection_matrix, cmap=plt.cm.BrBG, vmax=1, vmin=-1)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xticklabels(params)
ax.set_yticklabels(params)
for i in range(5):
    for ij in range(5):
        c = intersection_matrix[ij,i]
        ax.text(i, ij, str(c), va='center', ha='center')
        plt.title('Pearson correlation matrix PI-NN')
plt.savefig('{}/dependency_Pearson_corr_matrix_PINN'.format(pathresults))
plt.close('all')


#NNLS and LSQ
if arg.fit.do_fit:
    rhomatrix = [[1,round(matlsq[3][3],2),round(matlsq[1][4],2),round(matlsq[2][4],2),round(matlsq[0][4],2)],
                 [round(matlsq[3][3],2),1,round(matlsq[4][3],2),round(matlsq[0][3],2),round(matlsq[3][4],2)],
                 [round(matlsq[1][4],2),round(matlsq[4][3],2),1,round(matlsq[4][4],2),round(matlsq[1][3],2)],
                 [round(matlsq[2][4],2),round(matlsq[0][3],2),round(matlsq[4][4],2),1,round(matlsq[2][3],2)],
                 [round(matlsq[0][4],2),round(matlsq[3][4],2),round(matlsq[1][3],2),round(matlsq[2][3],2),1]]     
    fig, ax = plt.subplots()
    params = ['Dpar', 'Dint', 'Dmv', 'fint','fmv']
    intersection_matrix = np.array(rhomatrix)
    ax.matshow(intersection_matrix, cmap=plt.cm.BrBG, vmax=1, vmin=-1)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(params)
    ax.set_yticklabels(params)
    for i in range(5):
        for ij in range(5):
            c = intersection_matrix[ij,i]
            ax.text(i, ij, str(c), va='center', ha='center')    
    plt.title('Pearson correlation matrix LSQ')
    plt.savefig('{}/dependency_Pearson_corr_matrix_LSQ'.format(pathresults))
    plt.close('all')

    rhomatrix = [[1,round(matnnls[3][3],2),round(matnnls[1][4],2),round(matnnls[2][4],2),round(matnnls[0][4],2)],
                 [round(matnnls[3][3],2),1,round(matnnls[4][3],2),round(matnnls[0][3],2),round(matnnls[3][4],2)],
                 [round(matnnls[1][4],2),round(matnnls[4][3],2),1,round(matnnls[4][4],2),round(matnnls[1][3],2)],
                 [round(matnnls[2][4],2),round(matnnls[0][3],2),round(matnnls[4][4],2),1,round(matnnls[2][3],2)],
                 [round(matnnls[0][4],2),round(matnnls[3][4],2),round(matnnls[1][3],2),round(matnnls[2][3],2),1]]  
    fig, ax = plt.subplots()
    intersection_matrix = np.array(rhomatrix)
    ax.matshow(intersection_matrix, cmap=plt.cm.BrBG, vmax=1, vmin=-1)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])    
    ax.set_xticklabels(params)
    ax.set_yticklabels(params)
    for i in range(5):
        for ij in range(5):
            c = intersection_matrix[ij,i]
            ax.text(i, ij, str(c), va='center', ha='center')    
    plt.title('Pearson correlation matrix NNLS')
    plt.savefig('{}/dependency_Pearson_corr_matrix_NNLS'.format(pathresults))
    plt.close('all')
