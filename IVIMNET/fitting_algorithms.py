"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Built on code by Sebastiano Barbieri: https://github.com/sebbarb/deep_ivim


Revised January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component IVIM fitting in cerebrovascular disease (2022)

requirements:
numpy
tqdm
scipy
joblib
"""

# load relevant libraries
from scipy.optimize import curve_fit, nnls
import numpy as np
from joblib import Parallel, delayed
import tqdm
import warnings
from hyperparams import hyperparams as hp_example

# load hyperparameters
arg = hp_example()
#arg = checkarg_lsq(arg)



def fit_dats(bvalues, dw_data, arg, method, IR=True):
    """
    Wrapper function that selects the right fit depending on the method selected.
    input:
    :param arg: an object with fit options, with attributes:
    arg.do_fit --> Boolean; False for skipping the regular fit
    arg.fitS0 --> Boolean; False fixes S0 to 1, True fits S0 
    arg.jobs --> Integer specifying the number of parallel processes used in fitting. If <2, regular fitting is used instead
    arg.bounds --> 2D Array of fit bounds ([S0min, Dparmin, Fintmin, Dintmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fintmax, Dintmax, Fmvmax, Dmvmax]) 
    :param bvalues: 1D Array of b-values used
    :param dw_data: 2D Array containing the dw_data used with dimensions voxels x b-values
    :param method: string with the fit method; allowed: 'two-step-lsq' or 'NNLS'
    optional:
    :param IR: Boolean; True will fit the IVIM accounting for inversion recovery, False will fit IVIM without IR
    :return paramslsq: 2D array containing the fit parameters Dpar, Fint, Dint, Fmv, Dmv, and optionally, S0, for each voxel
    """
    # Checking completeness of arg and adding missing values as defaults
    arg = checkarg_lsq(arg)
    if arg.do_fit:
        # select fit to be run
        if (method == 'two-step-lsq'):
            print('running conventional fit\n')
            paramslsq = fit_least_squares_array_tri_exp(bvalues, dw_data, IR, njobs=arg.jobs, fitS0=arg.fitS0, bounds=arg.bounds)
        elif (method == 'NNLS'):
            print('running NNLS fit\n')
            paramslsq = fit_NNLS(bvalues, dw_data, IR = IR, bounds=arg.bounds)
        else:
            raise Exception('the choise lsq-fit is not implemented. Try ''two-step-lsq'' or ''NNLS''')
        return paramslsq
    # if fit is skipped, we return nothing
    return None


def tri_expN_noS0_IR(bvalues, Dpar, Fint, Dint, Fmv, Dmv):
    """ tri-exponential IVIM function accounted for inversion recovery (IR), and S0 set to 1"""
    return (( (1 - Fmv  - Fint ) * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.tissueT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.tissueT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.tissueT2-bvalues * Dpar ))
                           + Fint  * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.isfT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.isfT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.isfT2-bvalues * Dint ))
                           + Fmv  * ( (1 - np.exp(-arg.rel_times.repetitiontime/arg.rel_times.bloodT1)) * (np.exp(-arg.rel_times.echotime/arg.rel_times.bloodT2 -bvalues * (Dmv ) )) ))
                           / ( (1 - Fmv  - Fint ) * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.tissueT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.tissueT1) ) * np.exp(-arg.rel_times.echotime/arg.rel_times.tissueT2) 
                           + Fint  * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.isfT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.isfT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.isfT2))
                           + Fmv  * (1 - np.exp(-arg.rel_times.repetitiontime/arg.rel_times.bloodT1)) * np.exp(-arg.rel_times.echotime/arg.rel_times.bloodT2 )))

def tri_expN_IR(bvalues, S0, Dpar, Fint, Dint, Fmv, Dmv):
    """ tri-exponential IVIM function accounted for inversion recovery (IR)"""
    return (S0 * (( (1 - Fmv - Fint) * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.tissueT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.tissueT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.tissueT2-bvalues * Dpar))
                           + Fint * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.isfT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.isfT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.isfT2-bvalues * Dint))
                           + Fmv * ( (1 - np.exp(-arg.rel_times.repetitiontime/arg.rel_times.bloodT1)) * (np.exp(-arg.rel_times.echotime/arg.rel_times.bloodT2 -bvalues * (Dmv) )) ))
                           / ( (1 - Fmv - Fint) * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.tissueT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.tissueT1) ) * np.exp(-arg.rel_times.echotime/arg.rel_times.tissueT2) 
                           + Fint * (1 - 2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.isfT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.isfT1) ) * (np.exp(-arg.rel_times.echotime/arg.rel_times.isfT2))
                           + Fmv * (1 - np.exp(-arg.rel_times.repetitiontime/arg.rel_times.bloodT1)) * np.exp(-arg.rel_times.echotime/arg.rel_times.bloodT2 ))))

def tri_expN_noS0(bvalues, Dpar, Fint, Dint, Fmv, Dmv):
    """ tri-exponential IVIM function, and S0 set to 1"""
    return Fmv * np.exp(-bvalues * Dmv) + Fint * np.exp(-bvalues * Dint) + (1 - Fmv - Fint) * np.exp(-bvalues * Dpar)
       
def tri_expN(bvalues, S0, Dpar, Fint, Dint, Fmv, Dmv):
    """ tri-exponential IVIM function"""
    return S0 * (Fmv * np.exp(-bvalues * Dmv) + Fint * np.exp(-bvalues * Dint) + (1 - Fmv - Fint) * np.exp(-bvalues * Dpar))
   


def fit_least_squares_array_tri_exp(bvalues, dw_data, IR=True, njobs=4, fitS0=True, bounds=([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]), cutoff=200):
    """
    This is the LSQ implementation for a tri-exp model, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
    Second, we fit the other parameters using all b-values, while setting Dpar from step 1 as upper bound and starting value. This fit
    is done on an array.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param IR: Boolean; True will fit the IVIM accounting for inversion recovery, False will fit IVIM without IR; default=True
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fintmin, Dintmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fintmax, Dintmax, Fmvmax, Dmvmax]). default: ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2])
    :param cutoff: cutoff b-value used in step 1 
    :return Dpar: 1D Array with Dpar in each voxel
    :return Fmv: 1D Array with Fmv in each voxel
    :return Dmv: 1D Array with Dmv in each voxel
    :return Dint: 1D Array with Dint in each voxel
    :return Fint: 1D Array with Fint in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # first we normalise the signal to S0
    S0 = np.mean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]
    # here we try parallel computing, but if fails, go back to computing one single core.
    single = False
    if njobs > 2:
        try:
            # define the parallel function
            def parfun(i):
                return fit_least_squares_tri_exp(bvalues, dw_data[i, :], IR=IR, S0_output=True, fitS0=fitS0, bounds=bounds)

            output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True))
            Dpar, Fint, Dint, Fmv, Dmv, S0 = np.transpose(output)
        except:
            # if fails, retry using single core
            single = True
    else:
        # or, if specified, immediately go to single core
        single = True
    if single:
        # initialize empty arrays
        Dint = np.zeros(len(dw_data))
        Dpar = np.zeros(len(dw_data))
        S0 = np.zeros(len(dw_data))
        Fint = np.zeros(len(dw_data))
        Dmv = np.zeros(len(dw_data))
        Fmv = np.zeros(len(dw_data))
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            # fill arrays with fit results on a per voxel base:
            Dpar[i], Fint[i], Dint[i], Fmv[i], Dmv[i], S0[i] = fit_least_squares_tri_exp(bvalues, dw_data[i, :],IR=IR, S0_output=True, fitS0=fitS0, bounds=bounds)
    return [Dpar, Fmv, Dmv, Dint, Fint, S0]


def fit_least_squares_tri_exp(bvalues, dw_data, IR=True, S0_output=False, fitS0=True,
                      bounds=([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]), cutoff=200):
    """
    This is the LSQ implementation for a tri-exp model, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
    Second, we fit the other parameters using all b-values, while setting Dpar from step 1 as upper bound and starting value.
    It fits a single curve
    :param bvalues: 1D Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal in different voxels at different b-values
    :param IR: Boolean; True will fit the IVIM accounting for inversion recovery, False will fit IVIM without IR; default = True
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = False
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = True
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fintmin, Dintmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fintmax, Dintmax, Fmvmax, Dmvmax]). Default: ([0, 0, 0, 0.005, 0, 0.06], [2.5, 0.005, 1, 0.06, 1, 0.5])
    :param cutoff: cutoff b-value used in step 1 
    :return S0: optional 1D Array with S0 in each voxel
    :return Dpar: scalar with Dpar of the specific voxel
    :return Fint: scalar with Fint of the specific voxel
    :return Dint: scalar with Dint of the specific voxel
    :return Fmv: scalar with Fmv of the specific voxel
    :return Dmv: scalar with Dmv of the specific voxel
    """
     
    try:
        def monofit(bvalues, Dpar):
             return np.exp(-bvalues * Dpar)
        
        high_b = bvalues[bvalues >= cutoff]
        high_dw_data = dw_data[bvalues >= cutoff]
        boundspar = ([0], [bounds[1][1]])
        params, _ = curve_fit(monofit, high_b, high_dw_data, p0=[(bounds[1][1]-bounds[0][1])/2], bounds=boundspar)
        Dpar1 = params[0]
        if IR:
            if not fitS0:
                bounds = ([bounds[0][1] , bounds[0][2] , bounds[0][3] , bounds[0][4] , bounds[0][5] ],
                          [Dpar1          , bounds[1][2] , bounds[1][3] , bounds[1][4] , bounds[1][5] ])      
                params, _ = curve_fit(tri_expN_noS0_IR, bvalues, dw_data, p0=[Dpar1, 0.0, (bounds[0][3]+bounds[1][3])/2, 0.05, (bounds[0][5]+bounds[1][5])/2], bounds=bounds)
                Dpar, Fint, Dint, Fmv, Dmv = params[0], params[1], params[2], params[3], params[4]
                #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
                if Fint < 1e-4:
                    Dint = float("NaN")
                if Fmv < 1e-4:
                    Dmv = float("NaN")
                
            else:
                boundsupdated = ([bounds[0][0] , bounds[0][1] , bounds[0][2] , bounds[0][3] , bounds[0][4] , bounds[0][5] ],
                          [bounds[1][0] , Dpar1 , bounds[1][2] , bounds[1][3] , bounds[1][4] , bounds[1][5] ])
                params, _ = curve_fit(tri_expN_IR, bvalues, dw_data, p0=[1, Dpar1, 0.0, (bounds[0][3]+bounds[1][3])/2, 0.05, (bounds[0][5]+bounds[1][5])/2], bounds=boundsupdated)
                S0 = params[0]
                Dpar, Fint, Dint, Fmv, Dmv = params[1] , params[2] , params[3] , params[4] , params[5] 
                #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
                if Fint < 1e-4:
                    Dint = float("NaN")
                if Fmv < 1e-4:
                    Dmv = float("NaN")   
        else:
            if not fitS0:
                bounds = ([bounds[0][1] , bounds[0][2] , bounds[0][3] , bounds[0][4] , bounds[0][5] ],
                          [Dpar1          , bounds[1][2] , bounds[1][3] , bounds[1][4] , bounds[1][5] ])      
                params, _ = curve_fit(tri_expN_noS0, bvalues, dw_data, p0=[Dpar1, 0.0, (bounds[0][3]+bounds[1][3])/2, 0.05, (bounds[0][5]+bounds[1][5])/2], bounds=bounds)
                Dpar, Fint, Dint, Fmv, Dmv = params[0], params[1], params[2], params[3], params[4]
                #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
                if Fint < 1e-4:
                    Dint = float("NaN")
                if Fmv < 1e-4:
                    Dmv = float("NaN")
                
            else:
                boundsupdated = ([bounds[0][0] , bounds[0][1] , bounds[0][2] , bounds[0][3] , bounds[0][4] , bounds[0][5] ],
                          [bounds[1][0] , Dpar1 , bounds[1][2] , bounds[1][3] , bounds[1][4] , bounds[1][5] ])
                params, _ = curve_fit(tri_expN, bvalues, dw_data, p0=[1, Dpar1, 0.0, (bounds[0][3]+bounds[1][3])/2, 0.05, (bounds[0][5]+bounds[1][5])/2], bounds=boundsupdated)
                S0 = params[0]
                Dpar, Fint, Dint, Fmv, Dmv = params[1] , params[2] , params[3] , params[4] , params[5] 
                #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
                if Fint < 1e-4:
                    Dint = float("NaN")
                if Fmv < 1e-4:
                    Dmv = float("NaN")     
                    
        if S0_output:
            return Dpar, Fint, Dint, Fmv, Dmv, S0
        else:
            return Dpar, Fint, Dint, Fmv, Dmv
    except:

        if S0_output:
            return 0, 0, 0, 0, 0, 0
        else:
            return 0, 0, 0, 0, 0


def fit_NNLS(bvalues, dw_data, IR=True,
                      bounds=([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2])):
    """
    This is an implementation of the tri-exponential fit. It fits a single curve
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param IR: Boolean; True will fit the IVIM accounting for inversion recovery, False will fit IVIM without IR correction. default = True
    :param bounds: Array with fit bounds ([fp0min, Dparmin, Fintmin, Dintmin, Fmvmin, Dmvmin],[fp0max, Dparmax, Fintmax, Dintmax, Fmvmax, Dmvmax]). Default: ([0, 0, 0, 0.005, 0, 0.06], [2.5, 0.005, 1, 0.06, 1, 0.5])
    :return Fp0: optional 1D Array with f0 in each voxel
    :return Dpar: 1D Array with Dpar in each voxel
    :return Fint: 1D Array with Fint in each voxel
    :return Dint: 1D Array with Dint in each voxel
    :return Fmv: 1D Array with the fraciton of signal for Dmv in each voxel
    :return Dmv: 1D Array with Dmv in each voxel
    """
    
            
    try:
        Dspace = np.logspace(np.log10(bounds[0][1]), np.log10(bounds[1][5]), num=200)
        Dbasis = np.exp(-np.kron(np.reshape(bvalues,[len(bvalues),1]), np.reshape(Dspace,[1,len(Dspace)])))

        Dint = np.zeros(len(dw_data))
        Dpar = np.zeros(len(dw_data))
        Fint = np.zeros(len(dw_data))
        Fmv = np.zeros(len(dw_data))
        Dmv = np.zeros(len(dw_data))
        S0 = np.zeros(len(dw_data))
        
        def find_idx_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        
        idx_parint = find_idx_nearest(Dspace,bounds[1][1])
        idx_intmv = find_idx_nearest(Dspace,bounds[1][3])
        
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            # fill arrays with fit results on a per voxel base:
            x,rnorm = nnls(Dbasis,dw_data[i,:]) # x contains the diffusion spectrum
            ampl_Dpar = np.sum(x[:idx_parint]) #summing the amplitudes within the Dpar range
            ampl_Dint = np.sum(x[idx_parint:idx_intmv]) #summing the amplitudes within the Dint range
            ampl_Dmv = np.sum(x[idx_intmv:]) #summing the amplitudes within the Dmv range
            if len(np.nonzero(x[:idx_parint])[0])>0: # Dpar exists
                avg_Dpar = np.sum(Dspace[np.nonzero(x[:idx_parint])] * x[np.nonzero(x[:idx_parint])])/ampl_Dpar;
            else:
                avg_Dpar = 0
            if len(np.nonzero(x[idx_parint:idx_intmv])[0])>0: # Dint exists 
                avg_Dint = np.sum(Dspace[idx_parint:][np.nonzero(x[idx_parint:idx_intmv])] * x[idx_parint:][np.nonzero(x[idx_parint:idx_intmv])])/ampl_Dint;
            else:
                avg_Dint = 0
            if len(np.nonzero(x[idx_intmv:])[0])>0: # Dmv exists
                avg_Dmv = np.sum(Dspace[idx_intmv:][np.nonzero(x[idx_intmv:])] * x[idx_intmv:][np.nonzero(x[idx_intmv:])])/ampl_Dmv;
            else:
                avg_Dmv = 0
            
            if IR:
                corr_Fpar, corr_Fint, corr_Fmv = correct_for_IR(ampl_Dpar, ampl_Dint, ampl_Dmv)
                Fint[i] = corr_Fint
                Fmv[i] = corr_Fmv
            else:
                Fint[i] = ampl_Dint
                Fmv[i] = ampl_Dmv
            #when the fraction of a compartment equals zero, the corresponding diffusivity is non-existing (=NaN)
            if avg_Dpar == 0:
                avg_Dpar = float("NaN")
            if avg_Dint == 0:
                avg_Dint = float("NaN")
            if avg_Dmv == 0:
                avg_Dmv = float("NaN")
                
            Dpar[i] = avg_Dpar
            Dint[i] = avg_Dint
            Dmv[i] = avg_Dmv
            S0[i] = ampl_Dpar+ampl_Dint+ampl_Dmv # This is the sum before correction
            #note that after correction for IR, the sum of fractions always equals 1

        return Dpar, Fmv, Dmv, Dint, Fint, S0

    except:
        return 0, 0, 0, 0, 0, 0

        
def correct_for_IR(ampl_Dpar, ampl_Dint, ampl_Dmv):
    """
    This function corrects for the inversion recovery in the IVIM sequence, as described in Wong et al. (2019) Spectral Diffusion analysis of intravoxel incoherent motion MRI in cerebral small vessel disease
    :param ampl_Dpar: Scalar, the sum of amplitudes within the Dpar range
    :param ampl_Dint: Scalar, the sum of amplitudes within the Dint range
    :param ampl_Dmv: Scalar, the sum of amplitudes within the Dmv range
    :return corr_Fpar: Scalar, the fraction of Dpar, corrected for inversion recovery
    :return corr_Fint: Scalar, the fraction of Dint, corrected for inversion recovery
    :return corr_Fmv: Scalar, the fraction of Dmv, corrected for inversion recovery

    """
    TtLt = np.exp(-arg.rel_times.echotime/arg.rel_times.tissueT2)*(1-2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.tissueT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.tissueT1));
    TbLb = np.exp(-arg.rel_times.echotime/arg.rel_times.bloodT2)*(1-np.exp(-arg.rel_times.repetitiontime/arg.rel_times.bloodT1));
    TpLp = np.exp(-arg.rel_times.echotime/arg.rel_times.isfT2)*(1-2*np.exp(-arg.rel_times.inversiontime/arg.rel_times.isfT1) + np.exp(-arg.rel_times.repetitiontime/arg.rel_times.isfT1));

    #if all three components are present: 
    if ampl_Dpar>0 and ampl_Dint>0 and ampl_Dmv>0:
        #Calculate corrected fractions
        n1 = ((TbLb*ampl_Dpar)/(ampl_Dmv*TtLt))+1;
        n2 = (TtLt*TbLb*ampl_Dpar*ampl_Dint)/(ampl_Dpar*ampl_Dmv*TtLt*TpLp);
        denom = n1 + n2;
        z = 1/denom; # z is the microvascular fraction
        x = ((TbLb*ampl_Dpar)/(ampl_Dmv*TtLt))*z; # x is the parenchymal fraction
        y = 1-x-z; # y is the interstitial fluid fraction
        corr_Fpar = x;
        corr_Fint = y;
        corr_Fmv = z;   
        
    #if two components are present: 
    elif ampl_Dpar>0 and ampl_Dint>0 and ampl_Dmv==0:
        corr_Fint = 1/(((ampl_Dpar/ampl_Dint)*(TpLp/TtLt))+1);
        corr_Fpar = 1-corr_Fint;
        corr_Fmv = ampl_Dmv;       
    elif ampl_Dpar>0 and ampl_Dint==0 and ampl_Dmv>0:
        corr_Fmv = 1/(((ampl_Dpar/ampl_Dmv)*(TbLb/TtLt))+1);
        corr_Fpar = 1-corr_Fmv;
        corr_Fint = ampl_Dint;     
    elif ampl_Dpar==0 and ampl_Dint>0 and ampl_Dmv>0:
        corr_Fmv = 1/(((ampl_Dint/ampl_Dmv)*(TbLb/TpLp))+1);
        corr_Fint = 1-corr_Fmv;  
        corr_Fpar = ampl_Dpar;  
        
    #if one component is present: 
    else:
        corr_Fmv = ampl_Dmv; 
        corr_Fint = ampl_Dint;
        corr_Fpar = ampl_Dpar; 

    return corr_Fpar, corr_Fint, corr_Fmv



def checkarg_lsq(arg):
    if not hasattr(arg, 'do_fit'):
        warnings.warn('arg.fit.do_fit not defined. Using default of True')
        arg.do_fit=True
    if not hasattr(arg, 'fitS0'):
        warnings.warn('arg.fit.fitS0 not defined. Using default of False')
        arg.fitS0=False
    if not hasattr(arg, 'jobs'):
        warnings.warn('arg.fit.jobs not defined. Using default of 4')
        arg.jobs = 4
    if not hasattr(arg, 'bounds'):
        warnings.warn('arg.fit.bounds not defined. Using default of ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2])')
        arg.bounds = ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]) # S0, Dpar, Fint, Dint, Fmv, Dmv
    return arg
