"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Adapted January 2022 by Paulien Voorter
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

# import libraries
import numpy as np
import IVIMNET.deep as deep
import IVIMNET.fitting_algorithms as fit
import time
import scipy.stats as scipy
import torch
from joblib import Parallel, delayed
import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# seed 
torch.manual_seed(0)
np.random.seed(0)


def sim(SNR, arg):
    """ This function defines how well the different fit approaches perform on simulated data. Data is simulated by
    randomly selecting a value of Dpar, fint, Dint, fmv and Dmv from a predefined distribution. The script calculates the random,
    systematic, root-mean-squared error (RMSE) and Spearman Rank correlation coefficient for each of the IVIM parameters.
    Furthermore, it calculates the stability of the neural network (ensemble) when trained multiple times.

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param arg: an object with simulation options. hyperparams.py gives most details on the object (and defines it),
    Relevant attributes are:
    arg.sim.sims = number of simulations to be performed (need a large amount for training)
    arg.sim.num_samples_eval = number of samples to evaluate (save time for lsq fitting)
    arg.sim.distribution = distribution from which the IVIM parameters are sampled
    arg.sim.repeats = number of times to repeat the training and evaluation of the network (to assess stability)
    arg.sim.bvalues: 1D Array of b-values used
    arg.fit contains the parameters regarding lsq fitting
    arg.train_pars and arg.net_pars contain the parameters regarding the neural network
    arg.sim.range influences the simulated range (depending on the chosen distribution) of Dpar, fint, Dint, Fvm, Dmv in a 2D array

    :return matlsq: 2D array containing the performance of the lsq fit (if enabled). The rows indicate Dpar, fint, Dint, Fvm, Dmv
    , whereas the colums give the mean input value, the random error and the systematic error
    :return matnnls: 2D array containing the performance of the nnls fit (if enabled). The rows indicate Dpar, fint, Dint, Fvm, Dmv
    , whereas the colums give the mean input value, the random error and the systematic error
    :return matNN: 2D array containing the performance of the NN. The rows indicate Dpar, fint, Dint, Fvm, Dmv
    , whereas the colums give the mean input value, the random error and the systematic error
    :return stability: a 1D array with the stability of Dpar, fint, Dint, Fvm, Dmv as a fraction of their mean value.
    Stability is only relevant for neural networks and is calculated from the repeated network training.
    """
    arg = deep.checkarg(arg)
    # this simulated the signal
    IVIM_signal_noisy, Dpar, fint, Dint, fmv, Dmv = sim_signal(SNR, arg.sim.bvalues, arg.sim.IR, sims=arg.sim.sims,
                                             distribution=arg.sim.distribution, Dparmin=arg.sim.range[0][0],
                                             Dparmax=arg.sim.range[1][0], fintmin=arg.sim.range[0][1],
                                             fintmax=arg.sim.range[1][1], Dintmin=arg.sim.range[0][2],
                                             Dintmax=arg.sim.range[1][2], fmvmin=arg.sim.range[0][3],
                                             fmvmax=arg.sim.range[1][3], Dmvmin=arg.sim.range[0][4],
                                             Dmvmax=arg.sim.range[1][4], rician=arg.sim.rician)

    # only remember the Dpar, fint, Dint, Dmv and fmv needed for evaluation
    Dpar = Dpar[:arg.sim.num_samples_eval]
    Dint = Dint[:arg.sim.num_samples_eval]
    fint = fint[:arg.sim.num_samples_eval]
    Dmv = Dmv[:arg.sim.num_samples_eval]
    fmv = fmv[:arg.sim.num_samples_eval]

    # prepare a larger array if we use an ensemble of networks
    if arg.sim.n_ensemble>1:
        paramsNN = np.zeros([arg.sim.repeats, arg.sim.n_ensemble, 6, arg.sim.num_samples_eval])
    else :
        paramsNN = np.zeros([arg.sim.repeats, 6, arg.sim.num_samples_eval])

    # if we are not skipping the network for evaluation
    if not arg.train_pars.skip_net:
        # loop over repeats
        for aa in range(arg.sim.repeats):
            #Each repeat has an ensemble of bb trained networks
            if arg.sim.n_ensemble>1:
                #train network and predict parameters with multiple processes
                if arg.sim.jobs>1:
                    def parfun(i):
                        net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                        return deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net, arg)
                    output = Parallel(n_jobs=arg.sim.jobs)(delayed(parfun)(i) for i in tqdm.tqdm(range(arg.sim.n_ensemble), position=0, leave=True))
                    for bb in range(arg.sim.n_ensemble):
                        paramsNN[aa,bb] = output[bb]
                #train network and predict parameters with one process
                else:
                    net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                    for bb in range(arg.sim.n_ensemble):
                        paramsNN[aa,bb] = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net, arg)
                if arg.train_pars.use_cuda:
                   torch.cuda.empty_cache()
            else: #no ensemble of networks  
                    start_time = time.time()
                    # train network
                    print('\nRepeat: {repeat}\n'.format(repeat=aa))
                    net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                    elapsed_time = time.time() - start_time
                    print('\ntime elapsed for PI-NN training: {}\n'.format(elapsed_time))
                    start_time = time.time()
                    # predict parameters
                    if arg.sim.repeats > 1:
                        paramsNN[aa] = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                         arg)
                    else:
                        paramsNN = deep.predict_IVIM(IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net,
                                                     arg)
                    elapsed_time = time.time() - start_time
                    print('\ntime elapsed for PI-NN inference: {}\n'.format(elapsed_time))
                    # remove network to save memory
                    del net
                    if arg.train_pars.use_cuda:
                        torch.cuda.empty_cache()
        if arg.sim.n_ensemble>1: # take the average of the predictions from multiple network instances
            paramsNN = np.mean(paramsNN, axis=1)       
        print('\nresults for PI-NN')
        plot_dependency_figs(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), np.squeeze(paramsNN), 'PINN')
        # if we repeat training, then evaluate stability
        if arg.sim.repeats > 1:
            matNN = np.zeros([arg.sim.repeats, 5, 5])
            for aa in range(arg.sim.repeats):
                # determine errors and Spearman Rank
                matNN[aa] = print_errors(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), paramsNN[aa], arg)
            matNN = np.mean(matNN, axis=0)
            # calculate Stability Factor
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 4, 3, 1, 2]] / [np.mean(Dpar), np.mean(fint), np.mean(Dint), np.mean(fmv), np.mean(Dmv)]
            # set paramsNN for the plots
            paramsNN_0 = paramsNN[0]
        else:
            matNN = print_errors(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), np.squeeze(paramsNN), arg)
            stability = np.zeros(5)
            paramsNN_0 = paramsNN

    else:
        # if network is skipped
        stability = np.zeros(5)
        matNN = np.zeros([5, 5])
        
### least squares ###
    if arg.fit.do_fit:
        start_time = time.time()
        method = 'two-step-lsq'
        paramsf = fit.fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit, method, arg.sim.IR)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for LSQ fit: {} seconds\n'.format(elapsed_time))
        print('results for fit LSQ')
        # determine errors and Spearman Rank coeff
        matlsq = print_errors(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), paramsf, arg)
        plot_dependency_figs(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), paramsf, method)
### Non-negative least squares ###
    if arg.fit.do_fit:
        start_time = time.time()
        method = 'NNLS'
        paramsnnls = fit.fit_dats(arg.sim.bvalues, IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.fit, method, arg.sim.IR)
        elapsed_time = time.time() - start_time
        print('\ntime elapsed for NNLS fit: {} seconds\n'.format(elapsed_time))
        print('results for fit NNLS')
        # determine errors and Spearman Rank coeff
        matnnls = print_errors(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), paramsnnls, arg)
        plot_dependency_figs(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), paramsnnls, method)
###########################################################################################
    if arg.fit.do_fit:
        if not arg.train_pars.skip_net:
            plot_pred_vs_true(np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv), np.squeeze(paramsNN_0), paramsf, paramsnnls)
        return matlsq, matNN, matnnls, stability
    else:
        # if lsq and nnls fit are skipped, don't export lsq and nnls results
        return matNN, stability


def sim_signal(SNR, bvalues, IR = True, sims=1000000, distribution = 'normal', Dparmin=0.0001, Dparmax=0.0015, fintmin = 0.0, fintmax = 0.40, Dintmin = 0.0015, Dintmax = 0.004, fmvmin=0.0, fmvmax=0.2, Dmvmin=0.004, Dmvmax=0.2,
               rician=False, state=123):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of Dpar, fint, Dint, fmv and Dmv from a predefined distribution.

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param bvalues: 1D Array of b-values used
    
    optional:
    :param IR: Boolean, True for IVIM with inversion recovery and false for IVIM without inversion recovery. Default=True
    :param sims: number of simulations to be performed (need a large amount for training). Default=1000000
    :param distribution: describes the distribution from which the IVIM parameters are sampled. Default='normal'
    :param Dparmin: minimal simulated Dpar . Default = 0.0001
    :param Dparmax: maximal simulated Dpar. Default = 0.0015
    :param fintmin: minimal simulated fint. Default = 0
    :param fintmax: maximal simulated fint. Default = 0.40
    :param Dintmin: minimal simulated Dint. Default = 0.0015
    :param Dintmax: maximal simulated Dint. Default = 0.004
    :param fmvmin: minimal simulated fmv. Default = 0
    :param Dmvmax: minimal simulated fmv. Default = 0.20
    :param Dmvmin: minimal simulated Dmv. Default = 0.004
    :param Dmvmax: minimal simulated Dmv. Default = 0.2
    :param rician: boolean giving whether Rician noise is used; default = False

    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    :return Dpar: 1D array with the used Dpar for simulations, sims long
    :return fint: 1D array with the used fint for simulations, sims long
    :return Dint: 1D array with the used Dint for simulations, sims long
    :return fmv: 1D array with the used fmv for simulations, sims long
    :return Dmv: 1D array with the used Dmv for simulations, sims long
    """

    # randomly select parameters from predefined range 
    rg = np.random.RandomState(state)
   
    if (distribution == 'uniform'):
        test = rg.uniform(0, 1, (sims, 1)) 
        Dpar = Dparmin + (test * (Dparmax - Dparmin))
        test = rg.uniform(0, 1, (sims, 1)) 
        fint = fintmin + (test * (fintmax - fintmin)) 
        test = rg.uniform(0, 1, (sims, 1)) 
        Dint = Dintmin + (test * (Dintmax - Dintmin))
        test = rg.uniform(0, 1, (sims, 1)) 
        fmv = fmvmin + (test * (fmvmax - fmvmin)) 
        test = rg.uniform(0, 1, (sims, 1)) 
        Dmv =  Dmvmin + (test * (Dmvmax - Dmvmin))
    elif (distribution == 'normal'):
        test = rg.standard_normal((sims, 1))
        Dpar = np.absolute((Dparmax + Dparmin)/2 + (test * (Dparmax - Dparmin)/6)) #make sure that we don't simulate a negative Dpar
        test = rg.standard_normal((sims, 1))
        fint = np.absolute((fintmax + fintmin)/2 + (test * (fintmax - fintmin)/6)) #make sure that we don't simulate a negative fraction
        test = rg.standard_normal((sims, 1))
        Dint = (Dintmax + Dintmin)/2 + (test * (Dintmax - Dintmin)/6)
        test = rg.standard_normal((sims, 1))
        fmv = np.absolute((fmvmax + fmvmin)/2 + (test * (fmvmax - fmvmin)/6)) #make sure that we don't simulate a negative fraction
        test = rg.standard_normal((sims, 1))
        Dmv = (Dmvmax + Dmvmin)/2 + (test * (Dmvmax - Dmvmin)/6)
    elif (distribution == 'normal-wide'):
        test = rg.standard_normal((sims, 1))
        Dpar = np.absolute((Dparmax + Dparmin)/2 + (test * (Dparmax - Dparmin)/4)) #make sure that we don't simulate a negative Dpar
        test = rg.standard_normal((sims, 1))
        fint = np.absolute((fintmax + fintmin)/2 + (test * (fintmax - fintmin)/4)) #make sure that we don't simulate a negative fraction
        test = rg.standard_normal((sims, 1))
        Dint = (Dintmax + Dintmin)/2 + (test * (Dintmax - Dintmin)/4)
        test = rg.standard_normal((sims, 1))
        fmv = np.absolute((fmvmax + fmvmin)/2 + (test * (fmvmax - fmvmin)/4)) #make sure that we don't simulate a negative fraction
        test = rg.standard_normal((sims, 1))
        Dmv = (Dmvmax + Dmvmin)/2 + (test * (Dmvmax - Dmvmin)/4)
    else:
        raise Exception('the choise lsq-fit is not implemented. Try ''uniform'' or ''normal'' or normal-wide')
            
    # initialise data array
    data_sim = np.zeros([len(Dpar), len(bvalues)])
    bvalues = np.array(bvalues)

    # loop over array to fill with simulated IVIM data
    if IR:
        for aa in range(len(Dpar)):
            data_sim[aa, :] = fit.tri_expN_IR(bvalues, 1, Dpar[aa][0],fint[aa][0], Dint[aa][0], fmv[aa][0], Dmv[aa][0])
    else:
        for aa in range(len(Dpar)):
            data_sim[aa, :] = fit.tri_expN(bvalues, 1, Dpar[aa][0],fint[aa][0], Dint[aa][0], fmv[aa][0], Dmv[aa][0])       

    # if SNR is set to zero, don't add noise
    if SNR > 0:
        # initialise noise arrays
        noise_imag = np.zeros([sims, len(bvalues)])
        noise_real = np.zeros([sims, len(bvalues)])
        # fill arrays
        for i in range(0, sims - 1):
            noise_imag[i,] = rg.normal(0, 1 / SNR, (1, len(bvalues)))
        if rician:
            # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim

    # normalise signal
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]
    return data_sim, Dpar, fint, Dint, fmv, Dmv


def print_errors(Dpar, fint, Dint, fmv, Dmv, params, arg):
    """ this function calculates and prints the random, systematic, root-mean-squared (RMSE) errors and Spearman Rank correlation coefficient"""
    
    print('\nDpar was found {} times out of {}, percentage not found: {}%'.format(np.count_nonzero(~np.isnan(params[0])), arg.sim.num_samples_eval, 100*(arg.sim.num_samples_eval-np.count_nonzero(~np.isnan(params[0])))/arg.sim.num_samples_eval ))
    print('Dint was found {} times out of {}, percentage not found: {}%'.format(np.count_nonzero(~np.isnan(params[3])), arg.sim.num_samples_eval, 100*(arg.sim.num_samples_eval-np.count_nonzero(~np.isnan(params[3])))/arg.sim.num_samples_eval ))
    print('Dmv was found {} times out of {}, percentage not found: {}%'.format(np.count_nonzero(~np.isnan(params[2])), arg.sim.num_samples_eval, 100*(arg.sim.num_samples_eval-np.count_nonzero(~np.isnan(params[2])))/arg.sim.num_samples_eval ))
    
    rmse_Dpar = np.sqrt(np.nanmean(np.square(np.subtract(Dpar, params[0]))))
    rmse_fmv = np.sqrt(np.nanmean(np.square(np.subtract(fmv, params[1]))))
    rmse_Dmv = np.sqrt(np.nanmean(np.square(np.subtract(Dmv, params[2]))))
    rmse_Dint = np.sqrt(np.nanmean(np.square(np.subtract(Dint, params[3]))))
    rmse_fint = np.sqrt(np.nanmean(np.square(np.subtract(fint, params[4]))))

    # initialise Spearman Rank matrix
    Spearman = np.zeros([10, 1])
    # calculate Spearman Rank correlation coefficient   
    Spearman[0, 0] = scipy.stats.spearmanr(params[3], params[4], nan_policy='omit')[0]  # rho of Dint-fint
    Spearman[1, 0] = scipy.stats.spearmanr(params[2], params[1], nan_policy='omit')[0]  # rho of Dmv-fmv
    Spearman[2, 0] = scipy.stats.spearmanr(params[4], params[1], nan_policy='omit')[0]  # rho of fint-fmv
    Spearman[3, 0] = scipy.stats.spearmanr(params[0], params[3], nan_policy='omit')[0]  # rho of Dpar-Dint
    Spearman[4, 0] = scipy.stats.spearmanr(params[3], params[2], nan_policy='omit')[0]  # rho of Dint-Dmv
    Spearman[5, 0] = scipy.stats.spearmanr(params[0], params[1], nan_policy='omit')[0]  # rho of Dpar-fmv
    Spearman[6, 0] = scipy.stats.spearmanr(params[0], params[2], nan_policy='omit')[0]  # rho of Dpar-Dmv
    Spearman[7, 0] = scipy.stats.spearmanr(params[0], params[4], nan_policy='omit')[0]  # rho of Dpar-fint
    Spearman[8, 0] = scipy.stats.spearmanr(params[1], params[3], nan_policy='omit')[0]  # rho of fmv-Dint
    Spearman[9, 0] = scipy.stats.spearmanr(params[2], params[4], nan_policy='omit')[0]  # rho of Dmv-fint

    # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman[np.isnan(Spearman)] = 1

    meanDpar_true = np.mean(Dpar)
    meanfint_true = np.mean(fint)
    meanDint_true = np.mean(Dint)
    meanfmv_true = np.mean(fmv)
    meanDmv_true = np.mean(Dmv)
    
    meanDpar_fitted = np.nanmean(params[0])
    meanfint_fitted = np.nanmean(params[4])
    meanDint_fitted = np.nanmean(params[3])
    meanfmv_fitted = np.nanmean(params[1])
    meanDmv_fitted = np.nanmean(params[2])
    del params

    print('\nresults: columns show the mean true param, mean fitted param, and the NRMSE \n'
          'the rows show Dpar, fint, Dint, fmv and Dmv\n')
    print([meanDpar_true, meanDpar_fitted, rmse_Dpar / meanDpar_true])
    print([meanfint_true, meanfint_fitted, rmse_fint / meanfint_true])
    print([meanDint_true, meanDint_fitted, rmse_Dint / meanDint_true])
    print([meanfmv_true, meanfmv_fitted, rmse_fmv / meanfmv_true])
    print([meanDmv_true, meanDmv_fitted, rmse_Dmv / meanDmv_true])

    mats = [[meanDpar_true, meanDpar_fitted, rmse_Dpar / meanDpar_true, Spearman[0, 0], Spearman[5, 0]],
            [meanfint_true, meanfint_fitted, rmse_fint / meanfint_true, Spearman[1, 0], Spearman[6, 0]],
            [meanDint_true, meanDint_fitted, rmse_Dint / meanDint_true, Spearman[2, 0], Spearman[7, 0]],
            [meanfmv_true, meanfmv_fitted, rmse_fmv / meanfmv_true, Spearman[3, 0], Spearman[8, 0]],
            [meanDmv_true, meanDmv_fitted, rmse_Dmv / meanDmv_true, Spearman[4, 0], Spearman[9, 0]]]

    return mats                 


def plot_pred_vs_true(Dpar, fint, Dint, fmv, Dmv, paramsNN, paramslsq, paramsnnls):
    """ plots to visualize accuracy of Dpar, fint, Dint, fmv, Dmv for PI-NN, LSQ and NNLS"""
    
    pathresults = '{folder}/results'.format(folder=os.getcwd())
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dpar, paramslsq[0],color='coral', linestyle='', marker='.', markersize=1)
    plt.plot(Dpar, paramsnnls[0],color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dpar, paramsNN[0],color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([.0001,.0015]),np.array([.0001,.0015]), 'k--')
    plt.xlabel(('true Dpar'))
    plt.ylabel(('estimated Dpar'))
    plt.axis((0.00003,0.00157,0.00003,0.00157))
    plt.legend(('LSQ', 'NNLS', 'PI-NN'))
    plt.show()
    plt.pause(0.001)   
    #save
    plt.savefig('{}/accuracy_Dpar.png'.format(pathresults))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fmv, paramslsq[1],color='coral', linestyle='', marker='.', markersize=1)
    plt.plot(fmv, paramsnnls[1],color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(fmv, paramsNN[1],color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([.000,.2]),np.array([.000,.2]), 'k--')
    plt.xlabel(('true fmv'))
    plt.ylabel(('estimated fmv'))
    plt.axis((-0.01,0.21,-0.01,0.21))
    plt.legend(('LSQ', 'NNLS', 'PI-NN'))
    plt.show()
    plt.pause(0.001)   
    #save
    plt.savefig('{}/accuracy_Fmv.png'.format(pathresults))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dmv, paramslsq[2],color='coral', linestyle='', marker='.', markersize=1)
    plt.plot(Dmv, paramsnnls[2],color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dmv, paramsNN[2],color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.004,.2]),np.array([0.004,.2]), 'k--')
    plt.xlabel(('true Dmv'))
    plt.ylabel(('estimated Dmv'))
    plt.axis((0.004-.0098,0.2+.0098,0.004-.0098,0.2+.0098))
    plt.legend(('LSQ', 'NNLS', 'PI-NN'))
    plt.show()
    plt.pause(0.001)   
    #save
    plt.savefig('{}/accuracy_Dmv.png'.format(pathresults))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dint, paramslsq[3],color='coral', linestyle='', marker='.', markersize=1)
    plt.plot(Dint, paramsnnls[3],color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dint, paramsNN[3],color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.0015,.004]),np.array([0.0015,.004]), 'k--')
    plt.xlabel(('true Dint'))
    plt.ylabel(('estimated Dint'))
    plt.axis((0.0015-.000125,0.004+.000125,0.0015-.000125,0.004+.000125))
    plt.legend(('LSQ', 'NNLS', 'PI-NN'))
    plt.show()
    plt.pause(0.001)   
    #save
    plt.savefig('{}/accuracy_Dint.png'.format(pathresults))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, paramslsq[4],color='coral', linestyle='', marker='.', markersize=1)
    plt.plot(fint, paramsnnls[4],color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(fint, paramsNN[4],color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.0,.4]),np.array([0.0,.4]), 'k--')
    plt.xlabel(('true Fint'))
    plt.ylabel(('estimated Fint'))
    plt.axis((-0.02,0.42,-0.02,0.42))
    plt.legend(('LSQ', 'NNLS', 'PI-NN'))
    plt.show()
    plt.pause(0.001)   
    #save
    plt.savefig('{}/accuracy_Fint.png'.format(pathresults))
    plt.close('all')
    
def plot_dependency_figs(D, fint, Dint, f, Dp, params, method):
    """visualizes the interdependency between the fitted parameters. (not explored in publication)"""

    pathresults = '{folder}/results'.format(folder=os.getcwd())

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, D, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[4], params[0], 'b.', markersize=1)
    plt.xlabel(('Fint'))
    plt.ylabel(('Dpar'))
    plt.legend(('ground truth', 'predicted'))
    plt.xlim(0,1)
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fint_D_{}.png'.format(pathresults, method))
    plt.close('all')

    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(f, D, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[1], params[0], 'b.', markersize=1)
    plt.xlabel(('Fmv'))
    plt.legend(('ground truth', 'predicted'))
    plt.xlim(0,1)
    plt.ylabel(('Dpar'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fp_D_{}.png'.format(pathresults, method))
    plt.close('all')

    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dp, D, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[2], params[0], 'b.', markersize=1)
    plt.xlabel(('Dmv'))
    plt.ylabel(('Dpar'))
    plt.legend(('ground truth', 'predicted'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Dp_D_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dint, D, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[3], params[0], 'b.', markersize=1)
    plt.xlabel(('Dint'))
    plt.ylabel(('Dpar'))
    plt.legend(('ground truth', 'predicted'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Dint_D_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(f, Dp, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[1], params[2], 'b.', markersize=1)
    plt.xlabel(('Fmv'))
    plt.legend(('ground truth', 'predicted'))
    plt.xlim(0,1)
    plt.ylabel(('Dmv'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fp_Dp_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(f, Dint, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[1], params[3], 'b.', markersize=1)
    plt.legend(('ground truth', 'predicted'))
    plt.xlabel(('Fmv'))
    plt.xlim(0,1)
    plt.ylabel(('Dint'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fp_Dint_{}.png'.format(pathresults, method))
    plt.close('all')


    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, f, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[4], params[1], 'b.', markersize=1)
    plt.xlabel(('Fint'))
    plt.xlim(0,1)
    plt.ylabel(('Fmv'))
    plt.legend(('ground truth', 'predicted'))
    plt.ylim(0,1)
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fint_Fp_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dint, Dp, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[3], params[2], 'b.', markersize=1)
    plt.xlabel(('Dint'))
    plt.ylabel(('Dmv'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Dint_Dp_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, Dp, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[4], params[2], 'b.', markersize=1)
    plt.xlabel(('Fint'))
    plt.xlim(0,1)
    plt.ylabel(('Dmv'))
    plt.legend(('ground truth', 'predicted'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fint_Dp_{}.png'.format(pathresults, method))
    plt.close('all')
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, Dint, color='lightgray', linestyle='', marker='.', markersize=2)
    plt.plot(params[4], params[3], 'b.', markersize=1)
    plt.xlabel(('Fint'))
    plt.xlim(0,1)
    plt.ylabel(('Dint'))
    plt.legend(('ground truth', 'predicted'))
    plt.show()
    plt.pause(0.001)    
    #save
    plt.savefig('{}/dependency_Fint_Dint_{}.png'.format(pathresults, method))
    plt.close('all')


