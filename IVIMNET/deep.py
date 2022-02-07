"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim


Adapted January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component IVIM fitting in cerebrovascular disease (2022)

requirements:
numpy
torch
tqdm
matplotlib
"""
# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import IVIMNET.fitting_algorithms as fit
import copy
import warnings

# Define the neural network.
class Net(nn.Module):
    def __init__(self, bvalues, net_pars, rel_times):
        """
        this defines the Net class which is the network we want to train.
        :param bvalues: a 1D array with the b-values
        :param net_pars: an object with network design options, as explained in the publication Kaandorp et al., with attributes:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        times len(bvalues), with data sorted per voxel. This option was not explored in the publication
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        batch_norm --> Boolean determining whether to use batch normalisation
        parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
        (True), or have them all estimated by a single network (False)
        con --> string which determines what type of constraint is used for the parameters. Options are:
        'sigmoid' allowing a sigmoid constraint
	    'sigmoidabs' allowing a sigmoid constraint for the diffusivities Dpar, Dint and Dmv, while constraining the corresponding fraction to be positive
        'abs' having the absolute of the estimated values to constrain parameters to be positive
        'none' giving no constraints
        cons_min --> 1D array, if sigmoid is the constraint, these values give [Dpar_min, Fint_min, Dint_min, Fmv_min, Dmv_min, S0_min]
        cons_max --> 1D array, if sigmoid is the constraint, these values give [Dpar_max, Fint_max, Dint_max, Fmv_max, Dmv_min, S0_max]
        depth --> integer giving the network depth (number of layers)
        :params rel_times: an object with relaxation times of compartments and acquisition parameters, which is needed to correct for inversion recovery
        bloodT2 --> T2 of blood
        tissueT2 --> T2 of parenchymal tissue
        isfT2 --> T2 of interstitial fluid
        bloodT1 --> T1 of blood
        tissueT1 --> T1 of parenchymal tissue
        isfT1--> T1 of interstitial fluid
        echotime 
        repetitiontime
        inversiontime
        """
        super(Net, self).__init__()
        self.bvalues = bvalues
        self.net_pars = net_pars
        self.rel_times = rel_times
        if self.net_pars.width is 0:
            self.net_pars.width = len(bvalues)
        # define number of parameters being estimated
        self.est_pars = 5
        if self.net_pars.fitS0:
            self.est_pars += 1
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers0 = nn.ModuleList()
        if self.net_pars.parallel:
            self.fc_layers1 = nn.ModuleList()
            self.fc_layers2 = nn.ModuleList()
            self.fc_layers3 = nn.ModuleList()
            self.fc_layers4 = nn.ModuleList()
            self.fc_layers5 = nn.ModuleList()
        # loop over the layers
        width = len(bvalues)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.fc_layers0.extend([nn.Linear(width, self.net_pars.width)])
            if self.net_pars.parallel:
                self.fc_layers1.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers4.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers5.extend([nn.Linear(width, self.net_pars.width)])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.batch_norm:
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers4.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers5.extend([nn.BatchNorm1d(self.net_pars.width)])
            # add ELU units for non-linearity
            self.fc_layers0.extend([nn.ELU()])
            if self.net_pars.parallel:
                self.fc_layers1.extend([nn.ELU()])
                self.fc_layers2.extend([nn.ELU()])
                self.fc_layers3.extend([nn.ELU()])
                self.fc_layers4.extend([nn.ELU()])
                self.fc_layers5.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout is not 0 and i is not (self.net_pars.depth - 1):
                self.fc_layers0.extend([nn.Dropout(self.net_pars.dropout)])
                if self.net_pars.parallel:
                    self.fc_layers1.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers4.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers5.extend([nn.Dropout(self.net_pars.dropout)])
        # Final layer yielding output, with either 5 (fix S0) or 6 outputs of a single network, or 1 output
        # per network in case of parallel networks.
        if self.net_pars.parallel:
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 1))
            self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 1))
            self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 1))
            self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1))
            self.encoder4 = nn.Sequential(*self.fc_layers4, nn.Linear(self.net_pars.width, 1))
            if self.net_pars.fitS0:
                self.encoder5 = nn.Sequential(*self.fc_layers5, nn.Linear(self.net_pars.width, 1))
        else:
            self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, self.est_pars))
    def forward(self, X):
        # select constraint method
        if self.net_pars.con == 'sigmoid':
            # define constraints
            Dparmin = self.net_pars.cons_min[0]
            Dparmax = self.net_pars.cons_max[0]
            fintmin = self.net_pars.cons_min[1]
            fintmax = self.net_pars.cons_max[1]
            Dintmin = self.net_pars.cons_min[2]
            Dintmax = self.net_pars.cons_max[2]
            fmvmin = self.net_pars.cons_min[3]
            fmvmax = self.net_pars.cons_max[3]
            Dmvmin = self.net_pars.cons_min[4]
            Dmvmax = self.net_pars.cons_max[4]
            S0min = self.net_pars.cons_min[5]
            S0max = self.net_pars.cons_max[5]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params0 = self.encoder0(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel:
                params1 = self.encoder1(X)
                params2 = self.encoder2(X)
                params3 = self.encoder3(X)
                params4 = self.encoder4(X)
                if self.net_pars.fitS0:
                    params5 = self.encoder5(X)
        elif self.net_pars.con == 'sigmoidabs':
            # define constraints
            Dparmin = self.net_pars.cons_min[0]
            Dparmax = self.net_pars.cons_max[0]
            Dintmin = self.net_pars.cons_min[2]
            Dintmax = self.net_pars.cons_max[2]
            Dmvmin = self.net_pars.cons_min[4]
            Dmvmax = self.net_pars.cons_max[4]
            S0min = self.net_pars.cons_min[5]
            S0max = self.net_pars.cons_max[5]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params0 = self.encoder0(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel:
                params1 = self.encoder1(X)
                params2 = self.encoder2(X)
                params3 = self.encoder3(X)
                params4 = self.encoder4(X)
                if self.net_pars.fitS0:
                    params5 = self.encoder5(X)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.con == 'abs':
                # this network constrains the estimated parameters to be positive by taking the absolute.
                # Advantage is that the parameters are constrained and that the derrivative of the function remains
                # constant. Disadvantage is that -x=x, so could become unstable.
                params0 = torch.abs(self.encoder0(X))
                if self.net_pars.parallel:
                    params1 = torch.abs(self.encoder1(X))
                    params2 = torch.abs(self.encoder2(X))
                    params3 = torch.abs(self.encoder3(X))
                    params4 = torch.abs(self.encoder4(X))
                    if self.net_pars.fitS0:
                        params5 = torch.abs(self.encoder5(X))
            else:
                # this network is not constraint
                params0 = self.encoder0(X)
                if self.net_pars.parallel:
                    params1 = self.encoder1(X)
                    params2 = self.encoder2(X)
                    params3 = self.encoder3(X)
                    params4 = self.encoder4(X)
                    if self.net_pars.fitS0:
                        params5 = self.encoder5(X)
        else:
            raise Exception('the chose parameter constraint is not implemented. Try ''sigmoid'', ''sigmoidabs'',''none'' or ''abs''')
        X_temp=[]
        if self.net_pars.con == 'sigmoid':
            # applying constraints
            if self.net_pars.parallel:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = fmvmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (fmvmax - fmvmin)
                Dint = Dintmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = fintmin + torch.sigmoid(params4[:, 0].unsqueeze(1)) * (fintmax - fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params5[:, 0].unsqueeze(1)) * (S0max - S0min)
            else:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = fmvmin + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (fmvmax - fmvmin)
                Dint = Dintmin + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = fintmin + torch.sigmoid(params0[:, 4].unsqueeze(1)) * (fintmax - fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (S0max - S0min)
        elif self.net_pars.con == 'sigmoidabs':
            # applying constraints
            if self.net_pars.parallel:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = torch.abs(params2[:, 0].unsqueeze(1))
                Dint = Dintmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = torch.abs(params4[:, 0].unsqueeze(1))
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params5[:, 0].unsqueeze(1)) * (S0max - S0min)
            else:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = torch.abs(params0[:, 2].unsqueeze(1))
                Dint = Dintmin + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = torch.abs(params0[:, 4].unsqueeze(1))
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (S0max - S0min)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.parallel:
                Dmv = params0[:, 0].unsqueeze(1)
                Dpar = params1[:, 0].unsqueeze(1)
                Fmv = params2[:, 0].unsqueeze(1)
                Dint = params3[:, 0].unsqueeze(1)
                Fint = params4[:, 0].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params5[:, 0].unsqueeze(1)
            else:
                Dmv = params0[:, 0].unsqueeze(1)
                Dpar = params0[:, 1].unsqueeze(1)
                Fmv = params0[:, 2].unsqueeze(1)
                Dint = params0[:, 3].unsqueeze(1)
                Fint = params0[:, 4].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params0[:, 5].unsqueeze(1)
        # here we estimate X, the signal as function of b-values given the predicted IVIM parameters. Although
        # this parameter is not interesting for prediction, it is used in the loss function
        if self.net_pars.IR:
            if self.net_pars.fitS0:
                X_temp.append( 
                    S0 * (( (1 - Fmv - Fint) * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.tissueT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.tissueT1) ) * (torch.exp(-self.rel_times.echotime/self.rel_times.tissueT2-self.bvalues * Dpar))
                               + Fint * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.isfT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.isfT1) ) * (torch.exp(-self.rel_times.echotime/self.rel_times.isfT2-self.bvalues * Dint))
                               + Fmv * ( (1 - np.exp(-self.rel_times.repetitiontime/self.rel_times.bloodT1)) * (torch.exp(-self.rel_times.echotime/self.rel_times.bloodT2 -self.bvalues * (Dmv) )) ))
                               / ( (1 - Fmv - Fint) * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.tissueT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.tissueT1) ) * np.exp(-self.rel_times.echotime/self.rel_times.tissueT2) 
                               + Fint * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.isfT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.isfT1) ) * (np.exp(-self.rel_times.echotime/self.rel_times.isfT2))
                               + Fmv * (1 - np.exp(-self.rel_times.repetitiontime/self.rel_times.bloodT1)) * np.exp(-self.rel_times.echotime/self.rel_times.bloodT2 ))))
                                
            else:
                X_temp.append( 
                    ( (1 - Fmv - Fint) * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.tissueT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.tissueT1) ) * (torch.exp(-self.rel_times.echotime/self.rel_times.tissueT2-self.bvalues * Dpar))
                               + Fint * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.isfT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.isfT1) ) * (torch.exp(-self.rel_times.echotime/self.rel_times.isfT2-self.bvalues * Dint))
                               + Fmv * ( (1 - np.exp(-self.rel_times.repetitiontime/self.rel_times.bloodT1)) * (torch.exp(-self.rel_times.echotime/self.rel_times.bloodT2 -self.bvalues * (Dmv) )) ))
                               / ( (1 - Fmv - Fint) * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.tissueT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.tissueT1) ) * np.exp(-self.rel_times.echotime/self.rel_times.tissueT2) 
                               + Fint * (1 - 2*np.exp(-self.rel_times.inversiontime/self.rel_times.isfT1) + np.exp(-self.rel_times.repetitiontime/self.rel_times.isfT1) ) * (np.exp(-self.rel_times.echotime/self.rel_times.isfT2))
                               + Fmv * (1 - np.exp(-self.rel_times.repetitiontime/self.rel_times.bloodT1)) * np.exp(-self.rel_times.echotime/self.rel_times.bloodT2 )))
        else: 
            if self.net_pars.fitS0:
                X_temp.append( 
                    S0 * ((1 - Fmv - Fint) * (torch.exp(-self.bvalues * Dpar))
                               + Fint * (torch.exp(-self.bvalues * Dint))
                               + Fmv * (torch.exp(-self.bvalues * Dmv )) ))
                                   
            else:
                X_temp.append( 
                    ((1 - Fmv - Fint) * (torch.exp(-self.bvalues * Dpar))
                               + Fint * (torch.exp(-self.bvalues * Dint))
                               + Fmv * (torch.exp(-self.bvalues * Dmv )) ))
            
        X = torch.cat(X_temp,dim=1)
        if self.net_pars.fitS0:
            return X, Dpar, Fmv, Dmv, Dint, Fint, S0
        else:
            return X, Dpar, Fmv, Dmv, Dint, Fint, torch.ones(len(Dpar))


def learn_IVIM(X_train, bvalues, arg, net=None):
    """
    This program builds a IVIM-NET network and trains it.
    :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication Kaandorp et al. --> check hyperparameters.py for
    options
    :param net: an optional input pre-trained network with initialized weights for e.g. transfer learning or warm start
    :return net: returns a trained network
    """

    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
    X_train = X_train / S0[:, None]
    np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
    # removing non-IVIM-like data; this often gets through when background data is not correctly masked
    # Estimating IVIM parameters in these data is meaningless anyways.
    X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
    X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
    X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
    X_train[X_train > 1.5] = 1.5

    # initialising the network of choice using the input argument arg
    if net is None:
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues, arg.net_pars, arg.rel_times).to(arg.train_pars.device)
        #pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #print('\n trainable parameters = {} \n'.format(pytorch_total_trainable_params))
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'L1':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
    # train loader loads the training data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
    trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time. We can use substantially more data per batch as we are not training.
    inferloader = utils.DataLoader(val_set,
                                   batch_size=32 * arg.train_pars.batch_size,
                                   shuffle=False,
                                   drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    prev_lr = 0
    final_model = copy.deepcopy(net.state_dict())
    torch.set_num_threads(1) #number of codes used for training one network instance
    if arg.sim.n_ensemble > 1: #when training multiple network instances in parallel processes
        ## Train
        for epoch in range(1000):
            # initialising and resetting parameters
            net.train()
            running_loss_train = 0.
            running_loss_val = 0.
            for i, X_batch in enumerate(trainloader, 0):
                if i > totalit:
                    # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                    break
                # zero the parameter gradients
                optimizer.zero_grad()
                # put batch on GPU if pressent
                X_batch = X_batch.to(arg.train_pars.device)
                ## forward + backward + optimize
                X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0pred = net(X_batch)
                # removing nans and too high/low predictions to prevent overshooting
                X_pred[isnan(X_pred)] = 0
                X_pred[X_pred < 0] = 0
                X_pred[X_pred > 3] = 3
                # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
                loss = criterion(X_pred, X_batch)
                # updating network
                loss.backward()
                optimizer.step()
                # total loss and determine max loss over all batches
                running_loss_train += loss.item()
            # after training, do validation in unseen data without updating gradients
            #print('\n validation \n')
            net.eval()
            # validation is always done over all validation data
            for i, X_batch in enumerate(inferloader, 0):
                optimizer.zero_grad()
                X_batch = X_batch.to(arg.train_pars.device)
                # do prediction, only look at predicted IVIM signal
                X_pred, _, _, _, _, _, _ = net(X_batch)
                X_pred[isnan(X_pred)] = 0
                X_pred[X_pred < 0] = 0
                X_pred[X_pred > 3] = 3
                # validation loss
                loss = criterion(X_pred, X_batch)
                running_loss_val += loss.item()
            # scale losses
            running_loss_train = running_loss_train / totalit
            running_loss_val = running_loss_val / batch_norm2
            # save loss history for plot
            loss_train.append(running_loss_train)
            loss_val.append(running_loss_val)
            # as discussed in Kaandorp et al., LR is important. This approach allows to reduce the LR if we think it is too
            # high, and return to the network state before it went poorly
            if arg.train_pars.scheduler:
                scheduler.step(running_loss_val)
                if optimizer.param_groups[0]['lr'] < prev_lr:
                    net.load_state_dict(final_model)
                prev_lr = optimizer.param_groups[0]['lr']
            # early stopping criteria
            if running_loss_val < best:
                #print("\n############### Saving good model ###############################")
                final_model = copy.deepcopy(net.state_dict())
                best = running_loss_val
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg.train_pars.patience:
                    break
            # plot loss and plot 4 fitted curves
            if epoch > 0:
                # plot progress and intermediate results (if enabled)
                plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)
                
    else:     
        for epoch in range(1000):
            # initialising and resetting parameters
            net.train()
            running_loss_train = 0.
            running_loss_val = 0.
            for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
                if i > totalit:
                    # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                    break
                # zero the parameter gradients
                optimizer.zero_grad()
                # put batch on GPU if pressent
                X_batch = X_batch.to(arg.train_pars.device)
                ## forward + backward + optimize
                X_pred, Dpar_pred, Fmv_pred, Dmv_pred, Dint_pred, Fint_pred, S0pred = net(X_batch)
                # removing nans and too high/low predictions to prevent overshooting
                X_pred[isnan(X_pred)] = 0
                X_pred[X_pred < 0] = 0
                X_pred[X_pred > 3] = 3
                # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
                loss = criterion(X_pred, X_batch)
                # updating network
                loss.backward()
                optimizer.step()
                # total loss and determine max loss over all batches
                running_loss_train += loss.item()
            # after training, do validation in unseen data without updating gradients
            #print('\n validation \n')
            net.eval()
            # validation is always done over all validation data
            for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
                optimizer.zero_grad()
                X_batch = X_batch.to(arg.train_pars.device)
                # do prediction, only look at predicted IVIM signal
                X_pred, _, _, _, _, _, _ = net(X_batch)
                X_pred[isnan(X_pred)] = 0
                X_pred[X_pred < 0] = 0
                X_pred[X_pred > 3] = 3
                # validation loss
                loss = criterion(X_pred, X_batch)
                running_loss_val += loss.item()
            # scale losses
            running_loss_train = running_loss_train / totalit
            running_loss_val = running_loss_val / batch_norm2
            # save loss history for plot
            loss_train.append(running_loss_train)
            loss_val.append(running_loss_val)
            # as discussed in Kaandorp et al., LR is important. This approach allows to reduce the LR if we think it is too
            # high, and return to the network state before it went poorly
            if arg.train_pars.scheduler:
                scheduler.step(running_loss_val)
                if optimizer.param_groups[0]['lr'] < prev_lr:
                    net.load_state_dict(final_model)
                prev_lr = optimizer.param_groups[0]['lr']
            # early stopping criteria
            if running_loss_val < best:
                #print("\n############### Saving good model ###############################")
                final_model = copy.deepcopy(net.state_dict())
                best = running_loss_val
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg.train_pars.patience:
                    break
     
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/{name}_fig_fit.png'.format(name=arg.save_name))
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/{name}_fig_train.png'.format(name=arg.save_name))
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return net


def load_optimizer(net, arg):
    if arg.net_pars.parallel:
        if arg.net_pars.fitS0:
            par_list = [{'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()},
                        {'params': net.encoder3.parameters()}, {'params': net.encoder4.parameters()}, 
                        {'params': net.encoder5.parameters()}]
        else:
            par_list = [{'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()},
                        {'params': net.encoder3.parameters()}, {'params': net.encoder4.parameters()}]
    else:
        par_list = [{'params': net.encoder0.parameters()}]
    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.train_pars.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
                                                         patience=round(arg.train_pars.patience / 2))
        return optimizer, scheduler
    else:
        return optimizer


def predict_IVIM(data, bvalues, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param net: the trained IVIM-NET network
    :param arg: an object with network design options, as explained in the publication Kaandorp et al. check hyperparameters.py for
    options
    :return param: returns the predicted parameters
    """
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    S0 = np.mean(data[:, bvalues == 0], axis=1).astype('<f')
    data = data / S0[:, None]
    np.delete(data, isnan(np.mean(data, axis=1)), axis=0)
    # skip nans.
    mylist = isnan(np.mean(data, axis=1))
    sels = [not i for i in mylist]
    # remove data with non-IVIM-like behaviour. Estimating IVIM parameters in these data is meaningless anyways.
    sels = sels & (np.percentile(data[:, bvalues < 50], 0.95, axis=1) < 1.3) & (
                np.percentile(data[:, bvalues > 50], 0.95, axis=1) < 1.2) & (
                       np.percentile(data[:, bvalues > 150], 0.95, axis=1) < 1.0)
    # we need this for later
    lend = len(data)
    data = data[sels]

    # tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dmv = np.array([])
    Dpar = np.array([])
    Fmv = np.array([])
    Dint = np.array([])
    Fint = np.array([])
    S0 = np.array([])
    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)
    # start predicting
    if arg.sim.n_ensemble > 1: #when training multiple network instances in parallel 
        with torch.no_grad():
            for i, X_batch in enumerate(inferloader, 0):
                X_batch = X_batch.to(arg.train_pars.device)
                # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
                X_pred, Dpart, Fmvt, Dmvt, Dintt, Fintt, S0t = net(X_batch)
                # Quick and dirty solution to deal with networks not predicting S0
                try:
                    S0 = np.append(S0, (S0t.cpu()).numpy())
                except:
                    S0 = np.append(S0, S0t)
                Dmv = np.append(Dmv, (Dmvt.cpu()).numpy())
                Dpar = np.append(Dpar, (Dpart.cpu()).numpy())
                Fmv = np.append(Fmv, (Fmvt.cpu()).numpy())
                Dint = np.append(Dint, (Dintt.cpu()).numpy())
                Fint = np.append(Fint, (Fintt.cpu()).numpy())
    else:
        # start predicting
        with torch.no_grad():
            for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
                X_batch = X_batch.to(arg.train_pars.device)
                # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
                X_pred, Dpart, Fmvt, Dmvt, Dintt, Fintt, S0t = net(X_batch)
                # Quick and dirty solution to deal with networks not predicting S0
                try:
                    S0 = np.append(S0, (S0t.cpu()).numpy())
                except:
                    S0 = np.append(S0, S0t)
                Dmv = np.append(Dmv, (Dmvt.cpu()).numpy())
                Dpar = np.append(Dpar, (Dpart.cpu()).numpy())
                Fmv = np.append(Fmv, (Fmvt.cpu()).numpy())
                Dint = np.append(Dint, (Dintt.cpu()).numpy())
                Fint = np.append(Fint, (Fintt.cpu()).numpy())
   
  
    # The 'abs' and 'none' constraint networks have no way of figuring out what is Dpar, Dint and Dmv a-priori. However, they do
    # tend to pick one output parameter for Dpar, Dint or Dmv consistently within the network. If the network has swapped Dpar, Dint and/or
    # Dmv, we swap them back here.
    if np.mean(Dmv) < np.mean(Dint):
            D2 = copy.deepcopy(Dmv)
            Dmv = copy.deepcopy(Dint)
            Dint = copy.deepcopy(D2)
            F2 = copy.deepcopy(Fmv)
            Fmv = copy.deepcopy(Fint)
            Fint = copy.deepcopy(F2)
    if np.mean(Dint) < np.mean(Dpar):
        D2 = copy.deepcopy(Dpar)
        Dpar = copy.deepcopy(Dint)
        Dint = copy.deepcopy(D2)
        Fint = 1 - Fint - Fmv
    if np.mean(Dmv) < np.mean(Dint):
            D2 = copy.deepcopy(Dmv)
            Dmv = copy.deepcopy(Dint)
            Dint = copy.deepcopy(D2)
            F2 = copy.deepcopy(Fmv)
            Fmv = copy.deepcopy(Fint)
            Fint = copy.deepcopy(F2)
    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    # estimates
    Dmvtrue = np.zeros(lend)
    Dpartrue = np.zeros(lend)
    Fmvtrue = np.zeros(lend)
    Dinttrue = np.zeros(lend)
    Finttrue = np.zeros(lend)
    S0true = np.zeros(lend)
    Dmvtrue[sels] = Dmv
    Dpartrue[sels] = Dpar
    Fmvtrue[sels] = Fmv
    Dinttrue[sels] = Dint
    Finttrue[sels] = Fint
    S0true[sels] = S0
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return [Dpartrue, Fmvtrue, Dmvtrue, Dinttrue, Finttrue, S0true]


def isnan(x):
    """ this program indicates what are NaNs  """
    return x != x


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    """ this program plots the progress of the training. It will plot the loss and validatin loss, as well as 4 IVIM curve
    fits to 4 data points from the input"""
    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]
    if arg.fig:
        #matplotlib.use('TkAgg')
        plt.close('all')
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
        axs[0, 0].plot(bvalues, X_pred.data[0])
        axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
        axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
        axs[1, 0].plot(bvalues, X_pred.data[1])
        axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
        axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
        axs[0, 1].plot(bvalues, X_pred.data[2])
        axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
        axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
        axs[1, 1].plot(bvalues, X_pred.data[3])
        axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
        plt.legend(('data', 'estimate from network'))
        for ax in axs.flat:
            ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        for ax in axs.flat:
            ax.label_outer()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.figure(2)
        plt.clf()
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.yscale("log")
        plt.xlabel('epoch #')
        plt.ylabel('loss')
        plt.legend(('training loss', 'validation loss (after training epoch)'))
        plt.ion()
        plt.show()
        plt.pause(0.001)


def checkarg_train_pars(arg):
    if not hasattr(arg,'optim'):
        warnings.warn('arg.train.optim not defined. Using default ''adam''')
        arg.optim = 'adam'  # these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
    if not hasattr(arg,'lr'):
        warnings.warn('arg.train.lr not defined. Using default value 0.0001')
        arg.lr = 0.0001  # this is the learning rate. adam needs order of 0.001; 
    if not hasattr(arg, 'patience'):
        warnings.warn('arg.train.patience not defined. Using default value 10')
        arg.patience = 10  # this is the number of epochs without improvement that the network waits untill determining it found its optimum
    if not hasattr(arg,'batch_size'):
        warnings.warn('arg.train.batch_size not defined. Using default value 128')
        arg.batch_size = 128  # number of datasets taken along per iteration
    if not hasattr(arg,'maxit'):
        warnings.warn('arg.train.maxit not defined. Using default value 500')
        arg.maxit = 500  # max iterations per epoch
    if not hasattr(arg,'split'):
        warnings.warn('arg.train.split not defined. Using default value 0.9')
        arg.split = 0.9  # split of test and validation data
    if not hasattr(arg,'load_nn'):
        warnings.warn('arg.train.load_nn not defined. Using default of False')
        arg.load_nn = False
    if not hasattr(arg,'loss_fun'):
        warnings.warn('arg.train.loss_fun not defined. Using default of ''rms''')
        arg.loss_fun = 'rms'  # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
    if not hasattr(arg,'skip_net'):
        warnings.warn('arg.train.skip_net not defined. Using default of False')
        arg.skip_net = False
    if not hasattr(arg,'use_cuda'):
        arg.use_cuda = torch.cuda.is_available()
    if not hasattr(arg, 'device'):
        arg.device = torch.device("cuda:0" if arg.use_cuda else "cpu")
    return arg


def checkarg_net_pars(arg):
    if not hasattr(arg,'dropout'):
        warnings.warn('arg.net_pars.dropout not defined. Using default value of 0.1')
        arg.dropout = 0.1  # 0.0/0.1 chose how much dropout one likes. 0=no dropout; internet says roughly 20% (0.20) is good, although it also states that smaller networks might desire smaller amount of dropout
    if not hasattr(arg,'batch_norm'):
        warnings.warn('arg.net_pars.batch_norm not defined. Using default of True')
        arg.batch_norm = True  # False/True turns on batch normalistion
    if not hasattr(arg,'parallel'):
        warnings.warn('arg.net_pars.parallel not defined. Using default of True')
        arg.parallel = True  # defines whether the network exstimates each parameter seperately (each parameter has its own network) or whether 1 shared network is used instead
    if not hasattr(arg,'con'):
        warnings.warn('arg.net_pars.con not defined. Using default of ''sigmoid''')
        arg.con = 'sigmoid'  # defines the constraint function; 'sigmoid' gives a sigmoid function giving the max/min; 'abs' gives the absolute of the output, 'none' does not constrain the output
    if not hasattr(arg,'cons_min'):
        warnings.warn('arg.net_pars.cons_min not defined. Using default values')
        arg.cons_min = [0.0001, 0.0, 0.0015, 0.0, 0.004, 0.9]  # Dpar, Fint, Dint, Fmv, Dmv, S0
    if not hasattr(arg,'cons_max'):
        warnings.warn('arg.net_pars.cons_max not defined. Using default values')
        arg.cons_max = [0.0015, 0.40, 0.004, 0.2, 0.2, 1.1]  # Dpar, Fint, Dint, Fmv, Dmv, S0
    if not hasattr(arg,'fitS0'):
        warnings.warn('arg.net_pars.S0 not defined. Using default of False')
        arg.fitS0 = False  # indicates whether to fix S0 to 1 (for normalised signals); I prefer fitting S0 as it takes along the potential error is S0.
    if not hasattr(arg,'IR'):
        warnings.warn('arg.net_pars.IR not defined. Using default of True')
        arg.IR = False  #accounting for inversion recovery, True=yes, False=no
    if not hasattr(arg,'depth'):
        warnings.warn('arg.net_pars.depth not defined. Using default value of 4')
        arg.depth = 4  # number of layers
    if not hasattr(arg, 'width'):
        warnings.warn('arg.net_pars.width not defined. Using default of number of b-values')
        arg.width = 0
    return arg
          
def checkarg_sim(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg.sim.bvalues not defined. Using default value of [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]')
        arg.bvalues = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg.sim.repeats not defined. Using default value of 1')
        arg.repeats = 1  # this is the number of repeats for simulations
    if not hasattr(arg, 'rician'):
        warnings.warn('arg.sim.rician not defined. Using default of False')
        arg.rician = False
    if not hasattr(arg, 'SNR'):
        warnings.warn('arg.sim.SNR not defined. Using default of [20]')
        arg.SNR = [20]
    if not hasattr(arg, 'sims'):
        warnings.warn('arg.sim.sims not defined. Using default of 100000')
        arg.sims = 100000
    if not hasattr(arg, 'num_samples_eval'):
        warnings.warn('arg.sim.num_samples_eval not defined. Using default of 100000')
        arg.num_samples_eval = 100000
    if not hasattr(arg, 'range'):
        warnings.warn('arg.sim.range not defined. Using default of ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2])')
        arg.range =  ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
    return arg


def checkarg(arg):
    if not hasattr(arg, 'fig'):
        arg.fig = False
        warnings.warn('arg.fig not defined. Using default of False')
    if not hasattr(arg,'net_pars'):
        warnings.warn('arg no net_pars. Using default initialisation')
        arg.net_pars=net_pars()
    if not hasattr(arg, 'train_pars'):
        warnings.warn('arg no train_pars. Using default initialisation')
        arg.train_pars = train_pars()
    if not hasattr(arg, 'sim'):
        warnings.warn('arg no sim. Using default initialisation')
        arg.sim = sim()
    if not hasattr(arg, 'fit'):
        warnings.warn('arg no lsq. Using default initialisation')
        arg.fit = lsqfit()
    if not hasattr(arg, 'rel_times'):
        warnings.warn('arg no rel_times. Using default initialisation')
        arg.rel_times = rel_times()
    return arg

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
        self.scheduler = True # as discussed in the article, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = False


class net_pars:
    def __init__(self,nets):
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
        self.jobs = 4 # number of cores used to train the network instances of the ensemble in parallel 
        self.IR = True #True for IR-IVIM, False for IVIM without inversion recovery
        self.rician = False # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
 
class rel_times:
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
