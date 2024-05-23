"""
# Python package for Phase Retrieval (phr) reconstructions.

Main phr Paper Reference: 
Riccardo Battistelli, Daniel Metternich, Michael Schneider, Lisa-Marie Kern, Kai Litzius, Josefin Fuchs, Christopher Klose, Kathinka Gerlinger, Kai Bagschik, Christian M. Günther, Dieter Engel, Claus Ropers, Stefan Eisebitt, Bastian Pfau, Felix Büttner, and Sergey Zayko, "Coherent x-ray magnetic imaging with 5 nm resolution," Optica 11, 234-237 (2024) 
https://doi.org/10.1364/OPTICA.505999

@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)

Other applications by us of a version of this code:
- https://doi.org/10.1038/s41586-022-05537-9
- https://doi.org/10.1103/PhysRevB.107.L220405
- 10.1107/s1600577522010347

Other Phase Retrieval references:
- partial coherence algorithm: https://doi.org/10.1038/ncomms1994
- RAAR algorithm: 10.1088/0266-5611/21/1/004



hot to use this library:
########################################################

#RECIPE DETAILS: (if RL_freqs>Nit, the full coherence is assumed)
Nits=[700,50,50, 700, 50, 50]
algorithms=["RAARi", "ER", "ER", "RAARi", "ER", "ER"] 
beta_modes=['arctan','const', 'const','arctan', 'const', 'const']
average_img=3
RL_freqs=[2e9,2e9,2e9,20,20,20]
RL_its=[0,0,0,50,50,50]
smth_func=1
hels=[0,0,1,0,0,1]

# INSERT IMAGES AND CCD MASKs
im=[pos2,neg2]
bsmask=[bsmask_p, bsmask_n]

# these will contain the solutions
retr=[0,0]
retr_pc=[0,0]


guess_temp=Startimage.copy()
gamma_temp=Startgamma.copy()

for i in range(len(Nits)):
    
    start_time = time.time()
    Nit=Nits[i]
    algorithm=algorithms[i]
    beta_mode=beta_modes[i]
    RL_freq=RL_freqs[i]
    RL_it=RL_its[i]
    hel=hels[i]
  
    # readapt the diffraction pattern to fill the defective pixels covered by the bsmask for the first time we do partially coherent phase retrieval 
    if (RL_freq<Nit):
        if (RL_freqs[i-1]>Nits[i-1]): 
            im[0]=(np.abs(retr[0])**2)*bsmask[0] + np.maximum(im[0],np.zeros(im[0].shape))*(1-bsmask[0])
            im[1]=(np.abs(retr[0])**2)*bsmask[1] + np.maximum(im[1],np.zeros(im[1].shape))*(1-bsmask[1])
            bsmask=[0,0]
            gamma_temp=Startgamma.copy()
    
    #reconstruct positive helicity
    guess, error, gamma, supportmask = PhR2.phr(im=im[hel], mask=supportmask,  bsmask=bsmask[hel], mode=algorithm,
                 beta_zero=0.5, Nit=Nit, beta_mode=beta_mode, gamma=gamma_temp, RL_freq=RL_freq, RL_it=RL_it,
                 Phase=guess_temp, average_img=average_img)

    # get the guess as a starting point for the new cycle, but only if the helicity is the correct one
    if (hel==hels[0]):
        guess_temp=guess*np.sqrt(np.sum(im[1-hel])/np.sum(im[hel]))

    try:
        if (hel==hels[0]):
            gamma_temp=gamma.copy()
    except:
        gamma_temp=None
    #store the results in the appropriate arrays
    if RL_freq>Nit:
        retr[hel]=guess.copy()
    if RL_freq<Nit:
        retr_pc[hel]=guess.copy()
    print("--- %s x%d - %0.1f s ---" % (algorithm,Nit,time.time() - start_time))
###############################################################
"""
###################################################################
########## EXTERNAL DEPENDENCIES ##########

# TODO: make the library work also without GPU, just using numpy (all the conversion from cparray to np arrays must be adapted)

import sys, os

# Data
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
# Use cupy if there is a gpu

try:
    import cupy as xp
    GPU = xp.is_available()
    from cupy.fft import fft2 as fft2
    from cupy.fft import ifft2 as ifft2
    from cupy.fft import fftshift as fftshift
    from cupy.fft import ifftshift as ifftshift

except ImportError:
    import numpy as xp
    GPU = False
    from scipy.fft import fft2 as fft2
    from scipy.fft import ifft2 as ifft2
    from scipy.fft import fftshift as fftshift
    from scipy.fft import ifftshift as ifftshift

'''
import numpy as xp
GPU = False
from scipy.fft import fft2 as fft2
from scipy.fft import ifft2 as ifft2
from scipy.fft import fftshift as fftshift
from scipy.fft import ifftshift as ifftshift
'''
########## INTERNAL DEPENDENCIES ##########

# self-written libraries


########## CORE FUNCTIONS ##########

def PhaseRtrv_init(im,mask, bsmask=0 ,Nit=500, beta_zero=0.5, beta_mode='const', Phase=None,gamma=None, smth_func=1, seed=False):
    '''
    Initialization for terative phase retrieval function
    INPUT:  im: ArrayLike
                intensity of far field hologram data
            mask: ArrayLike
                support matrix, the reconstruction can be !=0 where mask==1
            bsmask: ArrayLike
                binary matrix used to mask camera artifacts. bmask==1 in spurious pixels and bmask==0 for pristine pixels. If bmask==1, the pixel will be left floating during the phase retrieval process
            Nit: int
                number of steps
            beta_zero: float
                starting value of beta
            beta_mode: ArrayLike/string
                way to evolve the beta parameter (specified array, const, arctan)
            Phase: ArrayLike
                initial starting guess in Fourier space, if Phase==None it's going to be a random start
            gamma: ArrayLike
                starting guess for Mutual Coherence Function (MCF) in Fourier space, if gamma==None then we will start with the assumption of perfect coherence
            smth_func: ArrayLike
                is used in the Richardson-Lucy routine, multiplied to the MCF. Can be used to suppress high-q values of the matrix
            seed: Boolean,
                if True, the starting value will be random but always using the same seed for more reproducible retrieved images   
    Returns:
    -------
        im: ArrayLike (cupy)
            amplitude of far field hologram data
        mask: ArrayLike (cupy)
            support matrix, the reconstruction can be !=0 where mask==1
        bsmask: ArrayLike (cupy)
            binary matrix used to mask camera artifacts. bmask==1 in spurious pixels and bmask==0 for pristine pixels. If bmask==1, the pixel will be left floating during the phase retrieval process
        guess: ArrayLike (cupy)
            starting guess in fourier space
        gamma: ArrayLike (cupy)
            starting MCF in Fourier space
        convolved: ArrayLike (cupy)
            if the algorithm is partially coherent, the square root of the guess convolved with gamma.
            if the algorithm is fully coherent, the abolute value of gamma.
        prev: ArrayLike (cupy)
            array to be used for the first support projection
        smth_func: ArrayLike (cupy)
            is used in the Richardson-Lucy routine, multiplied to the MCF. Can be used to suppress high-q values of the matrix  
        Beta: ArrayLike (cupy)
            list of beta values as a function of iteration steps
        error_list: list
            Magnitude error in Fourier space, list (returned empty at this point)
     --------
    author: RB 2024
    '''    
    #setup Error lists
    error_list=xp.zeros(0)

    #get a mask for negative values of im
    if type(bsmask)==int:
        bsmask=np.zeros(im.shape)
    bsmask[im<0]=1
    ### MODIFICATION
    bsmask = bsmask.astype(complex)
    # ELIMINATE ZEROES FROM im AND DO THE SQRT TO GO FROM INTENSITY TO AMPLITUDE
    im=np.sqrt(np.maximum(im,np.zeros(im.shape)))
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
          
    if seed==True:
        pass
        #np.random.seed(0)
      
    #set starting Phase guess
    if type(Phase)==type(None):
        Phase=np.exp(1j * np.random.rand(im.shape[0],im.shape[1])*np.pi*2)
        Phase=(1-bsmask)*im * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    guess = (1-bsmask)*im * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    
    #shift everything to the corner
    bsmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    im=np.fft.fftshift(im)
    if type(gamma)!=type(None):
        gamma=np.abs(np.fft.fftshift(gamma))
        gamma/=np.sum((gamma))
        
    if GPU:
        # get everything into cp arrays
        bsmask=xp.asarray(bsmask)
        guess=xp.asarray(guess)
        mask=xp.asarray(mask)
        im=xp.asarray(im)
    
    
    if type(Phase)==type(None):
        smth_func=smth_func.copy()
    else:
        if GPU:
            smth_func=xp.asarray(smth_func)
    
    #set the temporary convolved and prev arrays
    if type(gamma)!=type(None):
        if GPU:
            gamma=xp.asarray(gamma)
        convolved=xp.sqrt(ifft2(fft2(xp.abs(guess)**2) * fft2(gamma)))
        prev=fft2((1-bsmask) *im/xp.sqrt(convolved)* guess + guess * bsmask)
    else:
        convolved=xp.abs(guess)
        prev=fft2((1-bsmask) *im* xp.exp(1j * xp.angle(guess)) + guess*bsmask)

    return im, mask, bsmask, guess, gamma, convolved, prev, smth_func, Beta, error_list

  
def phr(im,mask,bsmask=0,Nit=500,mode='ER',beta_zero=0.5, beta_mode='const',Phase=None, gamma=None, RL_freq=25, RL_it=20,smth_func=1,
       plot_every=2000, average_img=10, Fourier_last=True, SW_freq=1e10,SW_sigma_list=0, SW_thr_list=0,seed=False):
    '''
    Iterative phase retrieval function.
    if RL_freq>Nit performs regular, fully coherent phase retrieval
    if RL_freq<Nit performs partially coherent phase retrieval with Richardson-Lucy (RL) algorithm    
    INPUT:  im: ArrayLike
                far field hologram data
            mask: ArrayLike
                support matrix, the reconstruction can be !=0 where mask==1
            bsmask: ArrayLike
                binary matrix used to mask camera artifacts. bmask==1 in spurious pixels and bmask==0 for pristine pixels. Where [bmask==1], the pixel will be left floating during the phase retrieval process
            Nit: int
                total number of steps
            mode: string
                algorithm to be used: ER, SF, RAARi, RAAR, HIOs, HIO, OSS, CHIO, HPR. "RAARi" is a modified version of "RAAR" without any sign constraint, for complex numbers
            beta_zero: float
                starting value of beta
            beta_mode: ArrayLike/string
                way to evolve the beta parameter (specified array, const, arctan)
            Phase: ArrayLike
                initial starting guess in Fourier space, if Phase==None it's going to be a random start
            gamma: ArrayLike
                starting guess for Mutual Coherence Function (MCF) in Fourier space
                gamma==None if we have the assumption of perfect coherence
            RL_freq: int
                how often the RL algorithm is applied. if RL_freq>Nit the algorithm will be fully coherent
            RL_it: int
                number of steps for every RL subroutine
            smth_func: array
                is used in the Richardson-Lucy subroutine, it is multiplied to the MCF. Can be used to suppress high-q values of the MFC to avoid divergence
            plot_every: int
                how often you plot data during the retrieval process
            average_img: int
                number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Fourier_last: Boolean
                Do you want to apply Fourier constraint one last time before returning results?                
            SW_freq: int
                To be used in case of Shrink Wrap algorithm. How often the SW is applied. 
            SW_sigma_list: ArrayLike
                To be used in case of Shrink Wrap algorithm. List of sigma for SW smoothing
            SW_thr_list: ArrayLike
                To be used in case of Shrink Wrap algorithm. List of thresholds applied for SW thresholding
            seed: Boolean,
                if True, the starting value will be random but always using the same seed for more reproducible retrieved images 
    Returns:
    -------
        guess: ArrayLike
            reconstructed complex solution (Fourier space)
        error_list: list
            Magnitude error in Fourier space, list (returned empty at this point)
        guess: ArrayLike
            reconstructed complex MCF (Fourier space)
        gamma: ArrayLike
                final estimate of the Mutual Coherence Function (MCF) in Fourier space
                gamma=None if we have the assumption of full coherence        
        mask: ArrayLike
            Updated SW mask (evolved from supportmask)
     --------
    author: RB 2024
    '''    
    first_instance_error=False
    #making sure that if we decided for a fully coherent algorithm it will be fully coherent
    if RL_freq>Nit:
        gamma=None
        
    # INITIAL SETUP
    im, mask, bsmask, guess, gamma, convolved, prev, smth_func, Beta, error_list = PhaseRtrv_init(im,mask, bsmask ,Nit, beta_zero, beta_mode, Phase,gamma, smth_func, seed)

    for s in range(0,Nit):
        beta=Beta[s]

        # APPLY MAGNITUDE CONSTRAINT -> GO TO REAL SPACE
        inv = fft2(((1-bsmask) *im/convolved + bsmask)* guess)

        # APPLY REAL SPACE CONSTRAINT: SUPPORT PROJECTION
        inv=supp_proj(inv, prev, mode, mask, beta, Nit,s)
        prev=xp.copy(inv)
        
        # SHRINK_WRAP
        if (s>0) and ((s%SW_freq)==0):
            mask=ShrinkWrap(inv,mask,SW_sigma_list[s], SW_thr_list[s])
          
        # RL SUBROUTINE
        if s>2 and (s%RL_freq==0):
            gamma = RL(inv=inv,guess=guess,bsmask=bsmask,im=im, gamma=gamma, RL_it=RL_it, smth_func=smth_func)
            
        # GO TO FOURIER SPACE
        guess=ifft2(inv)
        
        # COMPUTE PARTIALLY COHERENT DIFFRACTION PATTERN BY CONVOLVING WITH MCF   
        #and doing the SQRT for practical purposes
        if RL_freq<Nit:
            convolved=xp.sqrt(ifft2(fft2(xp.abs(guess)**2) * fft2(gamma)))
        else:
            convolved=xp.abs(guess)
        
        # COMPUTE ERRORS
        
        if (s % plot_every == 0) or (s>= Nit-average_img*2):
            error = error_diffract( (1-bsmask) * xp.abs(im)**2,  (1-bsmask) * convolved**2)
            error_list=xp.append(error_list,error)
            
        # ORDER AND SAVE RECONTSTRUCTED GUESSES BASED ON BEST ERROR PERFORMANCE
        if (s>= Nit-2*average_img):
            if first_instance_error==False:
                Best_guess=xp.zeros((average_img,im.shape[0], im.shape[1]),dtype = 'complex_')
                Best_error=xp.zeros(average_img)+1e10
                first_instance_error=True
                if RL_freq<Nit:
                    Best_gamma=xp.zeros((average_img,im.shape[0], im.shape[1]),dtype = 'complex_')
                
            if error<=xp.amax(Best_error):
                j=xp.argmax(Best_error)
                if error<Best_error[j]:
                    Best_error[j] = xp.copy(error)
                    Best_guess[j,:,:]=xp.copy(guess)
                    if RL_freq<Nit:
                        Best_gamma[j,:,:]=xp.copy(gamma)                   

    # sum best guess images dividing them for the number of items in Best_guess that are different from 0
    guess=xp.sum(Best_guess,axis=0)/xp.sum(Best_error!=0)
    if RL_freq<Nit:
        gamma=xp.sum(Best_gamma,axis=0)/xp.sum(Best_error!=0)
    
    # APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        if RL_freq<Nit:
            guess = (1-bsmask) *im/xp.sqrt(ifft2(fft2(xp.abs(guess)**2) * fft2(gamma))) * guess + guess * bsmask
            if GPU:
                gamma = ifftshift(xp.asnumpy(gamma))
        else:
            guess = (1-bsmask) *im* xp.exp(1j * xp.angle(guess)) + guess*bsmask
         
    if GPU:
        # CONVERT BACK TO NUMPY
        guess=xp.asnumpy(guess)
        error_list=xp.asnumpy(error_list)
        mask=xp.asnumpy(mask)
    
    # SHIFT TO CORNER
    guess=ifftshift(guess)
    mask=ifftshift(mask)
    
    return guess, error_list, gamma, mask


########## HELPER FUNCTIONS ##########

def supp_proj(inv, prev, mode, mask, beta, Nit,s):
    '''
    Applies support Projection in phase retrieval loop.
    INPUT:  inv: ArrayLike
                current real space reconstruction, after Fourier Space contraint and before Real Space constraint
            prev: ArrayLike
                real space reconstruction from the previous step, after Fourier Space contraint and Real Space constraint
            mode: ArrayLike
                algorithm to be used: "ER", "SF", "RAARi", "RAAR", "HIOs", "HIO", "OSS", "CHIO", "HPR".
                "RAARi" is a modified version of "RAAR" without any sign constraint, for complex numbers
            mask: int
                supportmask
            beta: float
                current value of beta at the step this function is executed
            Nit: int
                total number of steps
            s: int
                current step
    Returns:
    -------
        inv: ArrayLike
            current real space reconstruction, after applying the Real Space constraint
     --------
    author: RB 2024
    '''           
    match mode:
        case 'ER':
            inv*=mask
        case 'SF':
            inv*=(2*mask-1)
        case 'RAARi':
            inv += beta*(prev - 2*inv)*(1-mask)
        case 'RAAR':
            inv += beta*(prev - 2*inv)*(1-mask) * (2*inv-prev<0)
        case 'HIOs':
            inv +=  (1-mask)*(prev - (beta+1) * inv)
        case 'HIO':
            inv += (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*(xp.real(inv)<0)
        case 'OSS':
            (l,n) = inv.shape
            alpha=0.4
            inv += (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*(xp.real(inv)<0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* xp.floor(s/Nit*10)/10
            smoothed= ifft2( w(inv.shape[0],inv.shape[1],alpha) * fft2(inv))          
            inv = (inv*mask + (1-mask)*smoothed)
        case 'CHIO':
            alpha=0.4
            inv = (prev-beta*inv) + mask*(xp.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (xp.real(-inv+alpha*prev)>=0)*(xp.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
        case 'HPR':
            inv +=  (1-mask)*(prev - (beta+1) * inv)+ mask*(prev - (beta+1) * inv)*(xp.real(prev-(beta-3)*inv)>0)
    return inv


def RL(inv,guess,bsmask,im,  gamma, RL_it, smth_func=1):
    '''
    Subroutine for Richardson Lucy algorithm to be used inside phr_RL
    reference: https://doi.org/10.1038/ncomms1994
    
    INPUT:  inv: ArrayLike
                current real space reconstruction, after Fourier Space contraint and before Real Space constraint
            guess: ArrayLike
                Fourier space guess from previous iteration
            bsmask: ArrayLike
                binary matrix used to mask camera artifacts. bmask==1 in spurious pixels and bmask==0 for pristine pixels. Where [bmask==1], the pixel will be left floating during the phase retrieval process
            im: ArrayLike
                far field hologram data
            gamma: ArrayLike
                current version of the MCF in Fourier Space
            RL_it: int
                supportmask
            smth_func:  ArrayLike
                is used in the Richardson-Lucy routine, multiplied to the MCF. Can be used to suppress high-q values of the matrix
    Returns:
    -------
        gamma: ArrayLike
            updated MCF function in Fourier Space
     --------
    author: RB 2024
    '''
    new_guess=ifft2(inv)
    Idelta=2*xp.abs(new_guess)**2-xp.abs(guess)**2
    convolved=ifft2(fft2(xp.abs(new_guess)**2) * fft2(gamma))
    Iexp=(1-bsmask) *xp.abs(im)**2 + convolved * bsmask
            
    Id_1 = fft2(Idelta[::-1,::-1])
    Id   = fft2(Idelta)
    
    for l in range(RL_it):
        Denom=((ifft2((fft2(gamma)*Id))).real)
        Denom[Denom<1]=1e10
        gamma = xp.abs(gamma*ifft2(Id_1*(fft2(Iexp/Denom))))
        gamma[smth_func<=1e-5]=np.average(gamma[smth_func<=1e-5])
        gamma[smth_func>1e-5]=(gamma*smth_func)[smth_func>1e-5]
        gamma/=xp.nansum((gamma))

    gamma/= xp.nansum(gamma)

    return xp.abs(gamma)

def w(npx,npy,alpha=0.1):
    '''
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: int
                number of pixels on the image
            alpha: float
                width of the gaussian 
    Returns:
    -------
        inv: ArrayLike
            gaussian matrix
    --------
    author: RB 2024
    '''
    Y,X = xp.meshgrid(xp.arange(npy),xp.arange(npx))
    k=(xp.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return fftshift(xp.exp(-0.5*(k/alpha)**2))

def error_diffract(guess, im):
    '''
    Error of retrieved data with respect to the experimental data, Fourier Space
    INPUT:  guess: ArrayLike
                complex reconstructed far field image (Fourier space)
            im: ArrayLike
                experimental diffraction pattern (Fourier space) 
    Returns:
    -------
        Error: float
            Error (in dB)
    --------
    author: RB 2024
    '''
    Error = xp.sum( (im-guess)**2 )/ xp.sum(im**2)
    Error=10*xp.log10(Error)
    return Error