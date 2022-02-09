"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Adapted January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component IVIM fitting in cerebrovascular disease (2022)

requirements:
numpy
torch
"""
import torch
import numpy as np


class train_pars:
    def __init__(self,nets):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.00003 # this is the learning rate.
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = True # as discussed in the publications Kaandorp et al, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = False


class net_pars:
    def __init__(self,nets):
        # select a network setting
        if (nets == 'brain3'):
            self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = True # False/True turns on batch normalistion
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoidabs' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output, 
            #'sigmoidabs' gives a sigmoid constrant for diffusivities and an abs costraint for the components' fractions
            #### only if sigmoid or sigmoidabs constraint is used!
            self.cons_min = [0.0001, 0.0, 0.0015, 0.0, 0.004, 0.9]  # Dpar, Fint, Dint, Fmv, Dmv, S0
            self.cons_max = [0.0015, 0.40, 0.004, 0.2, 0.2, 1.1]  # Dpar, Fint, Dint, Fmv, Dmv, S0
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.IR = True #accounting for inversion recovery, True=yes, False=no
            self.depth = 2 # number of layers
            self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        else:
            # chose wisely :)
            self.dropout = 0.1 #0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
            self.batch_norm = True # False/True turns on batch normalistion
            self.parallel = True # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
            self.con = 'sigmoidabs' # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output, 
            #'sigmoidabs' gives a sigmoid constrant for diffusivities and a abs costraint for the component fractions
            #### only if sigmoid or sigmoidabs constraint is used!
            self.cons_min = [0.0001, 0.0, 0.0015, 0.0, 0.004, 0.9]  # Dpar, Fint, Dint, Fmv, Dmv, S0
            self.cons_max = [0.0015, 0.40, 0.004, 0.2, 0.2, 1.1]  # Dpar, Fint, Dint, Fmv, Dmv, S0
            ####
            self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
            self.IR = True #accounting for inversion recovery, True=yes, False=no
            self.depth = 2 # number of layers
            self.width = 0 # new option that determines network width. Putting to 0 makes it as wide as the number of b-values
        boundsrange = .3 * (np.array(self.cons_max)-np.array(self.cons_min)) # ensure that we are on the most lineair bit of the sigmoid function
        self.cons_min = np.array(self.cons_min) - boundsrange
        self.cons_max = np.array(self.cons_max) + boundsrange


class lsqfit:
    def __init__(self):
        self.do_fit = True # skip lsq fitting
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 4 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]) # S0, Dpar, Fint, Dint, Fmv, Dmv

class sim:
    def __init__(self):
        self.bvalues = np.array([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]) # array of b-values
        self.SNR = 35 # the SNR to simulate at
        self.sims = 11500000 # number of simulations to run
        self.num_samples_eval = 10000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.distribution = 'normal' #Define distribution from which IVIM parameters are sampled. Try 'uniform', 'normal' or 'normal-wide'
        self.repeats = 1 # this is the number of repeats for simulations to assess the stability
        self.n_ensemble = 20 # this is the number of instances in the network ensemble
        self.jobs = 4 # number of processes used to train the network instances of the ensemble in parallel (advised when training on cpu)
        self.IR = True #True for IR-IVIM, False for IVIM without inversion recovery
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
 
class rel_times:
    """relaxation times and acquisition parameters, which are required when accounting for inversion recovery"""
    def __init__(self):
        self.bloodT2 = 275 #ms [Wong et al. JMRI (2020)]
        self.tissueT2 = 95 #ms [Wong et al. JMRI (2020)]
        self.isfT2 = 503 # ms [Rydhog et al Magn.Res.Im. (2014)]
        self.bloodT1 =  1624 #ms [Wong et al. JMRI (2020)]
        self.tissueT1 =  1081 #ms [Wong et al. JMRI (2020)]
        self.isfT1 =  1250 # ms [Wong et al. JMRI (2020)]
        self.echotime = 84 # ms
        self.repetitiontime = 6800 # ms
        self.inversiontime = 2230 # ms
        
class hyperparams:
    def __init__(self):
        self.fig = False # plot intermediate steps during training
        self.save_name = 'brain3' 
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit()
        self.sim = sim()
        self.rel_times = rel_times()
        

        
