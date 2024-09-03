
"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction

2020
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
from scipy.stats import linregress
import fth_reconstruction as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import disk
import h5py
import math
import cupy as cp
import cupyx as cx #.scipy.ndimage.convolve1d
import gc

from numba import jit

#############################################################
#       PHASE RETRIEVAL FUNCTIONS
#############################################################

def PhaseRtrv(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20,BS=[0,0,0],bsmask=0,real_object=False,average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = disk((BS[1], BS[0]), BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    #else:
        #print("using phase given")

    guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=(BSmask)
    guess_cp=(guess)
    mask_cp=(mask)
    diffract_cp=(diffract)
    
    Best_guess=np.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=np.zeros(average_img)
    
    #print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        update = (1-BSmask_cp) *diffract_cp* np.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp
        
        ###REAL SPACE###
        inv = np.fft.fft2(update)
        if real_object:
            inv=np.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* np.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= np.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * np.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*np.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + np.heaviside(np.real(-inv+alpha*prev),1)*np.heaviside(np.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*np.heaviside(np.real(prev-(beta-3)*inv),0)

                        
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        guess_cp=np.fft.ifft2(inv)
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(np.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=np.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp
                    
        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(np.abs(guess_cp),mask_cp)
            Error_supp_list.append(Error_supp)
        
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    guess_cp=np.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* np.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp
        #print("FINAL \n",cp.average(diffract**2), cp.average(cp.abs(guess_cp)**2))
    
    guess=(guess_cp)
    #print("FINAL \n",cp.average(diffract**2), np.average(np.abs(guess)**2))

    #return final image
    return  (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list

def PhaseRtrv_harmonics(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20, BS=[0,0,0], bsmask=0, real_object=False, average_img=10, Nmodes=2):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Nmodes: number of modes
            
            
    OUTPUT:  retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx =disk((BS[1], BS[0]), BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
        
        
    
    mask_centered,center = center_mask(mask)
    supportmask=np.ones((Nmodes,l,n))*mask_centered
    supportmask[0,:,:]=mask_centered.copy()
    
    for i in range(1,Nmodes):
        supportmask[i,:,:]=expand_mask(mask_centered, factor=i+1)
    
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan(np.minimum((step-Nit/2),500)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.random.rand(l,n)*np.pi*2 #-np.pi/2
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        Phase=np.angle(Phase0)

    guess = (1-BSmask)*diffract * np.exp(1j * Phase)+ Phase0*BSmask
    print(guess.shape)
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    supportmask=np.fft.fftshift(supportmask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(supportmask)
    diffract_cp=cp.asarray(diffract)
    
    Best_guess=cp.zeros((average_img,Nmodes,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    
    PHI=cp.zeros((Nmodes,l,n), dtype=np.complex) #numpy array containing modes
    
    #if the starting image has no modes
    if guess.ndim == diffract.ndim:
        for j in range(Nmodes):
            PHI[j,:,:]=guess_cp.copy()/(j+1)
    else:
        PHI=guess_cp.copy()


    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        Iest=cp.sqrt(cp.sum(cp.abs(PHI)**2,axis=0))
        
        PHI= (1-BSmask_cp) * diffract_cp/Iest*(PHI) + PHI*BSmask_cp
        #update = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp

        
        ###REAL SPACE###
        inv = cp.fft.fft2(PHI)
        
        if real_object:
            inv=cp.real(inv)
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* cp.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)

                        
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        PHI=cp.fft.ifft2(inv)
        
        
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(Iest*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:,:]=PHI

        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(Iest,mask_cp)
            Error_supp_list.append(Error_supp)
        
            print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    PHI=cp.sum(Best_guess,axis=0)/average_img
    
    PHI_np=cp.asnumpy(PHI)

    #return final image
    return np.fft.fftshift(PHI_np)[:,::-1,::-1], Error_diffr_list, Error_supp_list

def PhaseRtrv_modes(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20, BS=[0,0,0], bsmask=0, real_object=False, average_img=10, Nmodes=5):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Nmodes: number of modes
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = disk((BS[1], BS[0]), BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan(np.minimum((step-Nit/2),500)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.random.rand(l,n)*np.pi*2 #-np.pi/2
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        Phase=np.angle(Phase0)

    guess = (1-BSmask)*diffract * np.exp(1j * Phase)+ Phase0*BSmask
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    Best_guess=cp.zeros((average_img,Nmodes,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    
    PHI=cp.zeros((Nmodes,l,n), dtype=np.complex) #numpy array containing modes
    
    #if the starting image has no modes
    if guess.ndim == diffract.ndim:
        for j in range(Nmodes):
            PHI[j,:,:]=guess_cp.copy()/(j+1)
    else:
        PHI=guess_cp.copy()
    #BSmask_cp=
    

    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        Iest=cp.sqrt(cp.sum(cp.abs(PHI)**2,axis=0))
        
        PHI= (1-BSmask_cp) * diffract_cp/Iest*(PHI) + PHI*BSmask_cp
        #update = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp

        
        ###REAL SPACE###
        inv = cp.fft.fft2(PHI)
        
        if real_object:
            inv=cp.real(inv)
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* cp.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)
               
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        PHI=cp.fft.ifft2(inv)

        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(Iest*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:,:]=PHI

        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(Iest,mask_cp)
            Error_supp_list.append(Error_supp)
        
            print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    PHI=cp.sum(Best_guess,axis=0)/average_img
    
    PHI_np=cp.asnumpy(PHI)

    #return final image
    return np.fft.fftshift(PHI_np)[:,::-1,::-1], Error_diffr_list, Error_supp_list
    
#################
def PhaseRtrv_GPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20, bsmask=0,real_object=False,average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT:  retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    Error_supp=0

    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    #else:
    #    print("using phase given")

    guess = (1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    #print("new")
  
    #previous result
    prev = None
    
    #shift everything to the corner
    bsmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(bsmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    first_instance_error=False
    
    #initialize prev
    prev=cp.fft.fft2((1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp)
    
    #print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        #update = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
        
        ###REAL SPACE###
        #inv = cp.fft.fft2(update)
        
        inv=cp.fft.fft2( guess_cp*((1-BSmask_cp) *diffract_cp/np.abs(guess_cp) + BSmask_cp))
        
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(cp.real(inv)<0)#cp.heaviside(- cp.real(inv) ,0)
        elif mode=='OSS':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(cp.real(inv)<0)#cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv = (inv*mask_cp + (1-mask_cp)*smoothed)
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*(cp.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (cp.real(-inv+alpha*prev)>=0)*(cp.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
            
        elif mode=='HPR':
            alpha=0.4
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(cp.real(prev-(beta-3)*inv)>0)
            
        prev=cp.copy(inv)

        
        ### FOURIER SPACE ### 
        guess_cp=cp.fft.ifft2(inv)
                
        #COMPUTE ERRORS
        if s<=2 or s % plot_every == 0:
            
            Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list.append(Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=cp.zeros(average_img)
                first_instance_error=True
            
            #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
            Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list.append(Error_diffr)
            
            if Error_diffr<=cp.amax(Best_error):
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j] = cp.copy(Error_diffr)
                    Best_guess[j,:,:]=cp.copy(guess_cp)


    #sum best guess images
    guess_cp=cp.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
    
    guess=cp.asnumpy(guess_cp)

    #return final image
    return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list

from scipy import signal


def PhaseRtrv_with_RL(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, RL_freq=25, RL_it=20, Phase=0,seed=False,
       plot_every=20, bsmask=0, real_object=False, average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function, with GPU acceleration and Richardson Lucy algorithm
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MOI
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
 
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    Error_supp=0

    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
          
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    #print("using phase given")

    gamma=np.fft.fftshift(gamma)
    gamma_cp=cp.asarray(gamma)
    gamma_cp/=cp.sum((gamma_cp))
        
    #guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    guess= Phase.copy()
  
    #previous result
    prev = None
    
    #shift everything to the corner
    bsmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(bsmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    first_instance_error=False
    
    #convolved=FFTConvolve(cp.abs(guess_cp)**2,gamma_cp)
    convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(guess_cp)**2) * cp.fft.fft2(gamma_cp))
    
    #initialize prev
    prev=cp.fft.fft2((1-BSmask_cp) *diffract_cp/cp.sqrt(convolved)* guess_cp + guess_cp * BSmask_cp)

    
    #print('mode=',mode,'    beta_mode=',beta_mode)

    for s in range(0,Nit):

        beta=Beta[s]
        
        #update = ((1-BSmask_cp) *diffract_cp/cp.sqrt(convolved) + BSmask_cp)* guess_cp

        
        ###REAL SPACE###
        #inv = cp.fft.fft2(update)
        
        inv = cp.fft.fft2( ((1-BSmask_cp) *diffract_cp/cp.sqrt(convolved) + BSmask_cp)* guess_cp )

            
        #support Projection
        if mode=='ER':
            inv*=mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)#*(cp.real(inv)<0)#cp.heaviside(- cp.real(inv) ,0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)#*(cp.real(inv)<0)#cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*(cp.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (cp.real(-inv+alpha*prev)>=0)*(cp.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
            #+ cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(cp.real(prev-(beta-3)*inv)>0)
            #*cp.heaviside(cp.real(prev-(beta-3)*inv),0)
            
            
        prev=cp.copy(inv)
        
        ### FOURIER SPACE ### 
        new_guess=cp.fft.ifft2(inv)

        #RL algorithm
        if s>RL_freq and (s%RL_freq==0):
            
            #convolved=FFTConvolve(cp.abs(new_guess)**2,gamma_cp)
            convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(new_guess)**2) * cp.fft.fft2(gamma_cp))
            Idelta=2*cp.abs(new_guess)**2-cp.abs(guess_cp)**2
            I_exp=(1-BSmask_cp) *cp.abs(diffract_cp)**2 + convolved * BSmask_cp
            gamma_cp = RL( Idelta=Idelta,  Iexp = I_exp , gamma_cp=gamma_cp, RL_it=RL_it)

            
        guess_cp = new_guess.copy()
        convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(guess_cp)**2) * cp.fft.fft2(gamma_cp))
        #FFTConvolve(cp.abs(guess_cp)**2,gamma_cp)
        
            
        if s<=2 or s % plot_every == 0:
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list.append(Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_gamma=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=cp.zeros(average_img)
                first_instance_error=True
                
            #compute error to see if image has to end up among the best.
            #Add in Best_guess array to sum them up at the end
            
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list.append(Error_diffr)
            
            if Error_diffr<=cp.amax(Best_error):
                j=cp.argmax(Best_error)
                #if Error_diffr<Best_error[j]:
                Best_error[j] = cp.copy(Error_diffr)
                Best_guess[j,:,:]=cp.copy(guess_cp)
                Best_gamma[j,:,:]=cp.copy(gamma_cp)
                    

    #sum best guess images dividing them for the number of items in Best_guess that are different from 0
    guess_cp=cp.sum(Best_guess,axis=0)/cp.sum(Best_error!=0)
    gamma_cp=cp.sum(Best_gamma,axis=0)/cp.sum(Best_error!=0)
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp/cp.sqrt(FFTConvolve(cp.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp

    guess=cp.asnumpy(guess_cp)
    gamma = cp.asnumpy(gamma_cp)
    
    return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list, np.fft.ifftshift(gamma)
    

    
#########

def RL(Idelta, Iexp, gamma_cp, RL_it):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''

    Id_1 = cp.fft.fft2(Idelta[::-1,::-1])
    Id   = cp.fft.fft2(Idelta)
    for l in range(RL_it):
        
        gamma_cp = gamma_cp*cp.fft.ifft2(Id_1*cp.fft.fft2(Iexp/(cp.fft.ifft2(Id*cp.fft.fft2(gamma_cp)))))
        gamma_cp/=cp.sum((gamma_cp))
        
    gamma_cp/= cp.nansum(gamma_cp)
        
    return gamma_cp

def RL_old(Idelta, Iexp, gamma_cp, RL_it):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''
    
    for l in range(RL_it):
        
        gamma_cp = (gamma_cp * (FFTConvolve(Idelta[::-1,::-1], Iexp/(FFTConvolve(Idelta,gamma_cp)))))
        gamma_cp/=cp.sum((gamma_cp))
        
    gamma_cp/= cp.nansum(gamma_cp)
        
    return gamma_cp

##########
def FFTConvolve(in1, in2):
    
    #in1[in1==cp.nan]=0
    #in2[in2==cp.nan]=0
    #ret = cp.fft.ifft2(cp.fft.fft2(in1) * cp.fft.fft2(in2))
    
    return cp.fft.ifft2(cp.fft.fft2(in1) * cp.fft.fft2(in2))
    
    
    
#########################################################################
#    Amplitude retrieval
#########################################################################

def AmplRtrv_GPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,
       plot_every=20,ROI=[None,None,None,None],BS=[0,0,0],bsmask=0,average_img=10):
    
    '''
    Iterative phase retrieval function, with GPU acceleration, for clearing the hologram from camera artifacts using "Amplitude rerieval"
    Makes sure that the FTH reconstruction is nonzero only inside the given "mask" support, and that its diffraction pattern stays real and positive.
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zero. Must include all obj.holes and reference holes reconstruction. can be obtained by doing the |FFT(mask)|**2 of the mask used for normal phase retrieval, and applying a threshold
            mode: string defining the algorithm to use (ER, RAAR, HIO)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            plot_every: how often you plot data during the retrieval process
            ROI: region of interest of the obj.hole, useful only for real-time imaging during the phase retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved image, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''
    
    #set titles of plotted images
    
    fig, ax = plt.subplots(1,3)   
    

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = disk((BS[1], BS[0]), BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-500)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.random.rand(l,n)*np.pi*2 #-np.pi/2
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        Phase=np.angle(Phase0)

    guess = (1-BSmask)*diffract * np.exp(1j * Phase)+ Phase0*BSmask
  
    #previous result
    prev = None
    
    FTHrec=fth.reconstruct(diffract)
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    #mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    FTHrec_cp=cp.asarray(FTHrec)
    cpROI=cp.asarray(ROI)
    
    Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
       
        update = guess_cp
        
        inv = (cp.fft.ifft2((update))) #inv is the FTH reconstruction
        
        inv=cp.fft.ifftshift(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #FTH reconstruction condition
        if mode=='ER':
            inv=FTHrec_cp*mask_cp
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='RAAR':
            inv = FTHrec_cp + (1-mask_cp)*(beta*prev - 2*beta*FTHrec_cp)
            + (beta*prev -2*beta*FTHrec_cp)* mask_cp* cp.where(-2*FTHrec_cp+prev>0,1,0)
            #cp.heaviside(cp.real(-2*inv+prev),0)
        #elif mode=='OSS':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        #    #smooth region outside support for smoothing
        #    alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
        #    smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
        #    inv= mask_cp*inv + (1-mask_cp)*smoothed
        #elif mode=='CHIO':
        #    alpha=0.4
        #    inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
        #    + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        #elif mode=='HPR':
        #    alpha=0.4
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #    + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)

                         
        prev=inv.copy()
        
        inv=cp.fft.fftshift(inv)
        
         #apply real and positive diffraction pattern constraint
        guess_cp = np.maximum(cp.real( cp.fft.fft2(inv) ) , cp.zeros(guess_cp.shape))
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp

        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            clear_output(wait=True)
            
            Error_supp = Error_support(cp.abs(guess_cp),mask_cp)
            #Error_supp_list.append(Error_supp)
            
            #ax1.scatter(s,Error_diffr,marker='o',color='red')
            #ax1bis.scatter(s,Error_supp,marker='x',color='blue')
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            guessplot=np.fft.fftshift(cp.asnumpy(guess_cp))
            
            im=(np.fft.ifft2((guessplot)))
            
            im_real=np.real(im)
            mir, mar = np.percentile(im_real[ROI], (0,100))
            print(mir,mar)
            im_imag=np.imag(im)
            
            ax[0].imshow((guessplot), cmap='coolwarm')
            #ax3.imshow(im_abs, cmap='binary')

            real_detail=im_real.copy()
            imag_detail=im_imag.copy()
            ax[1].imshow(real_detail,vmin=mir,vmax=mar)
            ax[2].imshow(imag_detail, cmap='hsv',vmin=-cp.pi,vmax=cp.pi)

            display(plt.gcf())
        
            print(cp.sum(guess_cp),'#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    #guess_cp==cp.sum(Best_guess,axis=0)/average_img
    
    guess=cp.asnumpy(guess_cp)

    #return final image
    return np.fft.ifftshift(guess) , Error_diffr_list, Error_supp_list

#############################################################
#    FILTER FOR OSS
#############################################################
def W(npx,npy,alpha=0.1):
    '''
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: number of pixels on the image
            alpha: width of the gaussian 
            
    OUTPUT: gaussian matrix
    
    --------
    author: RB 2020
    '''
    Y,X = cp.meshgrid(cp.arange(npy),cp.arange(npx))
    k=(cp.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return cp.fft.fftshift(cp.exp(-0.5*(k/alpha)**2))

#############################################################
#    ERROR FUNCTIONS
#############################################################
def Error_diffract(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=(diffract-guess)**2
    Den=diffract**2
    Error = Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

def Error_diffract_cp(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=cp.abs(diffract-guess)**2
    Den=cp.abs(diffract)**2
    Error = Num.sum()/Den.sum()#cp.sum(Num)/cp.sum(Den)
    Error=10*cp.log10(Error)
    return Error

def Error_support(prev,mask):
    '''
    Error on the support of retrieved data. 
    INPUT:  prev: retrieved image
            mask: support mask
            
    OUTPUT: Error on the support, how much prev is outside of "mask"
    
    --------
    author: RB 2020
    '''
    Num=prev*(1-mask)**2
    Den=prev**2
    Error=Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

#############################################################
#    function for saving Hdf5 file
#############################################################

"""
functions to create and read hdf5 files.
groups will be converted to dictionaries, containing the data
supports nested dictionaries.

to create hdf file:
    create_hdf5(dict0,filename) where dict0 is the dictionary containing the data and filename the file name
to read hdf file:
    data=cread_hdf5(filename) data will be a dictionary containing all information in "filename.hdf5"
riccardo 2020

"""

def read_hdf5(filename, extension=".hdf5", print_option=True):
    
    f = h5py.File(filename+extension, 'r')
    dict_output = readHDF5(f, print_option = print_option, extension=extension)
    
    return dict_output
    
def readHDF5(f, print_option=True, extension=".hdf5", dict_output={}):
    
    for i in f.keys():
        
    
        if type(f[i]) == h5py._hl.group.Group:
            if print_option==True:
                print("### ",i)
                print("---")
            dict_output[i]=readHDF5(f[i],print_option=print_option,dict_output={})
            if print_option==True:
                print("---")
        
        else:
            dict_output[i]=f[i][()]
            if print_option==True:
                print("•",i, "                  ", type(dict_output[i]))
        
        
    return dict_output
    
def create_hdf5(dict0,filename, extension=".hdf5"):
    
    f=createHDF5(dict0,filename, extension=extension)
    f.close()


def createHDF5(dict0,filename, extension=".hdf5",f=None):
    '''creates HDF5 data structures strating from a dictionary. supports nested dictionaries'''
    print(dict0.keys())
    
#    try:
#        f = h5py.File(filename+ ".hdf5", "w")
#        print("ok")
#    except OSError:
#        print("could not read")
    
    if f==None:
         f = h5py.File(filename+ extension, "w")
    
    
    if type(dict0) == dict:
        
        for i in dict0.keys():
            
            print("create group %s"%i)
            print("---")
            print(i,",",type(dict0[i]))

            if type(dict0[i]) == dict:
                print('dict')
                grp=(f.create_group(i))
                createHDF5(dict0[i],filename,f=grp)
                
            elif type(dict0[i]) == np.ndarray:
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
                
            elif (dict0[i] != None):
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
            print("---")
    return f


#############################################################
#    save parameters
#############################################################


def save_reco_dict_to_hdf(fname, reco_dict):
    '''Saves a flat dictionary to a new hdf group in given file.
    
    Parameters
    ----------
    fname : str
        hdf file name
    reco_dict : dict
        Flat dictionary
    
    Returns
    -------
    grp : str
        Name of the new data group.
    -------
    author: dscran 2020
    '''
    with h5py.File(fname, mode='a') as f:
        i = 0
        while f'reco{i:02d}' in f:
            i += 1
        for k, v in reco_dict.items():
            f[f'reco{i:02d}/{k}'] = v
    return f'reco{i:02d}'

def read_hdf(fname):
    '''
    reads the latest saved parameters in the hdf file
    INPUT:  fname: path and filename of the hdf file
    OUtPUT: image numbers, retrieved_p, retrieved_n, recon, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    i = 0
    while f'reco{i:02d}' in f:
        i += 1
    i -= 1
    
    nobs_numbers = f[f'reco{i:02d}/entry numbers no beamstop'].value
    sbs_numbers = f[f'reco{i:02d}/entry numbers small beamstop'].value
    lbs_numbers = f[f'reco{i:02d}/entry numbers large beamstop'].value
    retrieved_holo_p = f[f'reco{i:02d}/retrieved hologram positive helicity'].value 
    retrieved_holo_n = f[f'reco{i:02d}/retrieeved hologram negative helicity'].value 
    prop_dist = f[f'reco{i:02d}/Propagation distance'].value
    phase = f[f'reco{i:02d}/phase'].value
    roi = f[f'reco{i:02d}/ROI coordinates'].value

    return (nobs_numbers, sbs_numbers, lbs_numbers, retrieved_holo_p, retrieved_holo_n, prop_dist, phase, roi)


#############################################################
#    Ewald sphere projection
#############################################################

from scipy.interpolate import griddata

def inv_gnomonic(CCD, center=None, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, method='cubic' , mask=None):
    '''
    Projection on the Ewald sphere for close CCD images. Only gets the new positions on the new projected array and then interpolates them on a regular matrix
    Input:  CCD: far-field diffraction image
            z: camera-sample distance,
            center_y,center_x: pixels in excess we want to add to the borders by zero-padding so that the projected image has existing pixels to use
            px_size: size of CCD pixels
    Output: Output: projected image
    
    -------
    author: RB Nov2020
    '''
    
    # we have to caculate all new angles
    
    #points coordinates positions
    z=experimental_setup['ccd_dist']

    if type(center)==type(None):
        center=np.array([CCD.shape[1]/2, CCD.shape[0]/2])


    print("center=",center, "z=",z )
    values=CCD.flatten()
    points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))).astype('float64')
    
    points[0,:]-=center[0]
    points[1,:]-=center[1]
    points*= experimental_setup['px_size']
    
    
    #points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))- CCD.shape[0]/2) * px_size

    points=points.T
        
    #now we have to calculate the new points
    points2=np.zeros(points.shape)
    points2[:,0]= z* np.sin( np.arctan( points[:,0] / np.sqrt( points[:,1] **2 + z**2 ) ) )
    points2[:,1]= z* np.sin( np.arctan( points[:,1] / np.sqrt( points[:,0] **2 + z**2 ) ) )

    
    CCD_projected = griddata(points2, values, points, method=method)
    
    CCD_projected = np.reshape(CCD_projected, CCD.shape)
    
    #makes outside from nan to zero
    CCD_projected=np.nan_to_num(CCD_projected, nan=0, posinf=0, neginf=0)
    

    return CCD_projected

############################################
## Fourier Ring Correlation
############################################

def FRC0(im1,im2,width_bin):
    '''
    implements Fourier Ring Correlation. (https://www.nature.com/articles/s41467-019-11024-z)
    Input:  im1,im2: two diffraction patterns with different sources of noise. Can also use same image twice, sampling only odd/even pixels
            width_bin: width of circles we will use to have our histogram
            
    Output: sum_num: array of all numerators value of correlation hystogram
            sum_den: array of all denominators value of correlation hystogram
    
    -------
    author: RB 2020
    '''
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])
    
    for i in range(Num_bins):
        annulus = np.zeros(shape)
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum( im1* np.conj(im2) * annulus )#np.sum( im1[np.nonzero(annulus)] * np.conj(im2[np.nonzero(annulus)]) )
        sum_den[i]=np.sqrt( np.sum(np.abs(im1)**2* annulus) * np.sum(np.abs(im2)**2* annulus) )
        
    return sum_num,sum_den

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def FRC(im1,im2,width_bin, center=0):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    if type(center)==int:
        center = np.array([shape[0]//2, shape[1]//2])
    
    FT1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im1)))
    FT2=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im2)))
    
    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        
        yy,xx=np.where()
        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum(FT1* np.conj(FT2)[yy_outer, xx_outer]) - np.sum(FT1* np.conj(FT2)[yy_inner, xx_inner])
        sum_den[i]=np.sqrt( (np.sum(np.abs(FT1[yy_outer, xx_outer])**2)-np.sum(np.abs(FT1[yy_inner, xx_inner])**2)) * np.sum(np.abs(FT2[yy_outer, xx_outer])**2- np.abs(FT2[yy_inner, xx_inner])**2) )
        
    return sum_num,sum_den

def FRC_1image(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even'''
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    sum_num=np.zeros((4,Num_bins))
    sum_den=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        sum_num[j,:],sum_den[j,:]=FRC(im[FT1st[j]],im[FT2nd[j]],width_bin)

    FRC_array=sum_num/sum_den        
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    

    
def FRC_GPU2(im1,im2,width_bin,center=0, start_Fourier=True):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC histogram array
    
    RB 2020'''
    

    
    shape=im1.shape
    shape=(int(np.ceil(np.sqrt(2)*shape[0])),int(np.ceil(np.sqrt(2)*shape[0])))
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=cp.zeros(Num_bins)
    sum_den=cp.zeros(Num_bins)
    
    if type(center)==int:
        center = np.array([im1.shape[0]//2, im1.shape[1]//2])
    
    if start_Fourier:
        FT1=cp.asarray(im1)
        FT2=cp.asarray(im2)
    else:
        im1_cp=cp.asarray(im1)
        im2_cp=cp.asarray(im2)
        FT1=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im1_cp)))
        FT2=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im2_cp)))

    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
        if i==0:
            yy_inner, xx_inner = disk((center[0], center[1]), 1)

        outer=np.zeros((yy_outer.shape[0],2))
        outer[:,0]=xx_outer.copy()
        outer[:,1]=yy_outer.copy()

        in_delete=np.amax(outer, axis=1)>=(im1.shape[0])
        outer=np.delete(outer, in_delete, axis=0)
        in_delete=np.amin(outer, axis=1)<(0)
        outer=np.delete(outer, in_delete, axis=0)

        inner=np.zeros((yy_inner.shape[0],2))
        inner[:,0]=xx_inner.copy()
        inner[:,1]=yy_inner.copy()

        in_delete=np.amax(inner, axis=1)>=(im1.shape[0])
        inner=np.delete(inner, in_delete, axis=0)
        in_delete=np.amin(inner, axis=1)<(0)
        inner=np.delete(inner, in_delete, axis=0)

        inner=np.rint(inner).astype(int)
        outer=np.rint(outer).astype(int)
        sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[yy_outer, xx_outer] ) - cp.sum( (FT1* cp.conj(FT2))[yy_inner, xx_inner] )
        sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT1[yy_inner, xx_inner])**2)) * (cp.sum(cp.abs(FT2[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT2[yy_inner, xx_inner])**2)) )
        
    FRC_array=sum_num/sum_den
    FRC_array_np=cp.asnumpy(FRC_array)
    
    return FRC_array_np
    
def FRC_GPU(im1,im2,width_bin,center=0, start_Fourier=True):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC histogram array
    
    RB 2020'''
    

    
    shape=im1.shape
    #shape=(int(np.ceil(np.sqrt(2)*shape[0])),int(np.ceil(np.sqrt(2)*shape[0])))
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=cp.zeros(Num_bins)
    sum_den=cp.zeros(Num_bins)
    
    if type(center)==int:
        center = np.array([shape[0]//2, shape[1]//2])
    
    if start_Fourier:
        FT1=cp.asarray(im1)
        FT2=cp.asarray(im2)
    else:
        im1_cp=cp.asarray(im1)
        im2_cp=cp.asarray(im2)
        FT1=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im1_cp)))
        FT2=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im2_cp)))

    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
        sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[yy_outer, xx_outer] ) - cp.sum( (FT1* cp.conj(FT2))[yy_inner, xx_inner] )
        sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT1[yy_inner, xx_inner])**2)) * (cp.sum(cp.abs(FT2[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT2[yy_inner, xx_inner])**2)) )
        
    FRC_array=sum_num/sum_den
    FRC_array_np=cp.asnumpy(FRC_array)
    
    return FRC_array_np


def FRC_1image_GPU(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even
    
    RB 2020'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    FRC_array=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        FRC_array[j,:]=FRC_GPU(im[FT1st[j]],im[FT2nd[j]],width_bin)
      
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    
def half_bit_thrs(shape, SNR=0.5, width_bin=5):
    '''van heel and schatz 2005
    gives you an array containing values for the half bit threshold
    RB 2020'''

    Num_bins=shape[0]//(2*width_bin)
    center = np.array([shape[0]//2, shape[1]//2])
    thr=np.zeros(Num_bins)
    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0
        n=np.sum(annulus) #counting...
        #print(n)
    
        thr[i]=(SNR+ (2*np.sqrt(SNR)+1)/np.sqrt(n))/(SNR+1+2*np.sqrt(SNR)/np.sqrt(n))
    return thr

def PRTF(im, exp_im, width_bin=5):
    '''function for Phase Retrieval Transfer Function
    RB Jan 2021
    INPUT: im: sums of retrieved image
            exp_im: experimental scattering pattern
            width_bin: width of bins used to plot PRTF
    output: prtf: phase retrieval transfer function'''
    
    prtf= im/exp_im
    
    prtf_cp=cp.asarray(prtf)
    
    shape=prtf.shape
    Num_bins=shape[0]//(2*width_bin)
    
    prtf_array=cp.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])

    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        prtf_array[i]=cp.sum( prtf_cp * annulus )/cp.sum(annulus)
        
    prtf_array_np=cp.asnumpy(prtf_array)
    
    return prtf_array_np

def azimutal_integral_GPU_2(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked. FOR NOW IT DOES NOT WORK
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''

    if type(mask) is int:
        mask_cp=mask
        
    elif type(mask) is np.ndarray and mask.ndim==2:
        mask_cp=cp.asarray(mask) 
    
    
    if type(im) is np.ndarray and im.ndim==2:
        print("array 2D -- new with mask")
        shape=im.shape 
        center = cp.array([shape[0]//2, shape[1]//2])
        y, x = cp.indices((im.shape))
        r = cp.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(cp.int)
        print(r,r.shape)
        r[mask_cp==0]=shape[0]*3 # just to exclude them
        im_cp=cp.asarray(im)

         
        Num_bins=int(((np.sqrt(2)*shape[0])//(2*width_bin)))
        azimuthal_integral=cp.zeros(Num_bins)
                
        for i in cp.arange(Num_bins):
            azimuthal_integral[i]=cp.average(im_cp[(r>=i*width_bin) & (r<(i+1)*width_bin)])

            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            print("list")
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[cp.zeros(Num_bins)]*Num_images

        else: #list or 2D np array
            print("array 3D")
            im_cp=cp.asarray(im)
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=cp.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])

        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            
            if i%100==0:
                print("bin=",i)
            for j in range(Num_images):
                azimuthal_integral[j][i]=(cp.sum( im_cp[j][yy_outer,xx_outer]) -cp.sum(im_cp[j][yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
                
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np

def azimutal_integral_GPU(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked. FOR NOW IT DOES NOT WORK
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''
    
    if type(mask) is int:
        mask_cp=mask
        
    elif type(mask) is np.ndarray and mask.ndim==2:
        mask_cp=cp.asarray(mask) 
    
    
    if type(im) is np.ndarray and im.ndim==2:
        print("array 2D")
        im_cp=cp.asarray(im)
        shape=im.shape  
        Num_bins=shape[0]//(2*width_bin)
        azimuthal_integral=cp.zeros(Num_bins)
        
        center = np.array([shape[0]//2, shape[1]//2])
        #annulus = cp.zeros(shape)
        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            #annulus[yy_outer,xx_outer]=1
            #annulus[yy_inner,xx_inner]=0
            #annulus*=mask_cp

            #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
            #azimuthal_integral[i]=cp.sum( im_cp * annulus ) / cp.sum( annulus )
            azimuthal_integral[i]=(cp.sum( im_cp[yy_outer,xx_outer])- cp.sum(im_cp[yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            print("list")
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[cp.zeros(Num_bins)]*Num_images

            

        else: #list or 2D np array
            print("array 3D")
            im_cp=cp.asarray(im)
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=cp.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])
        #annulus = cp.zeros(shape)
        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            #annulus[yy_outer,xx_outer]=1
            #annulus[yy_inner,xx_inner]=0
            
            if i%100==0:
                print("bin=",i)
            for j in range(Num_images):
                #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
                #azimuthal_integral[j][i]=cp.sum( im_cp[j] * annulus*mask_cp ) / cp.sum( annulus*mask_cp )
                azimuthal_integral[j][i]=(cp.sum( im_cp[j][yy_outer,xx_outer]) -cp.sum(im_cp[j][yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
                
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np

def azimutal_integral(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''

    
    
    if type(im) is np.ndarray and im.ndim==2:
        print("array 2D")

        shape=im.shape  
        Num_bins=shape[0]//(2*width_bin)
        
        center = np.array([shape[0]//2, shape[1]//2])
        
        azimuthal_integral=radial_profile(im, center)
        
            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            print("list")
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
                
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[np.zeros(Num_bins)]*Num_images

            

        else: #list or 2D np array
            print("array 3D")
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=np.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])

        for j in range(Num_images):
            azimuthal_integral[j]=radial_profile(im[j], center)
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np



def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile





########
# WIDGET MASK SELECTION
########

import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
    
    
def holomask(holo, plot_max=94, RHdict={}):

    #DEFINE FUNCTION
    def add_on_button_clicked(b):
        
        x1, x2 = ax.get_xlim()
        y2, y1 = ax.get_ylim()

        # obj position
        x = fth.integer(x1 + (x2 - x1)/2)
        y = fth.integer(y1 + (y2 - y1)/2)
        #object radius
        r = fth.integer(np.maximum((x2 - x1)/2, (y2 - y1)/2))
        
        RHn=len(RHdict.keys())+1

        RHdict.update({"RH%d"%RHn: {"#":RHn,"r":r,"x":x,"y":y}})

        ax.set_xlim(0,holo.shape[1])
        ax.set_ylim(holo.shape[0],0)
        
        for i in RHdict:
            yy, xx = disk((i["y"],i["x"]),i["r"])

        
    fig, ax = plt.subplots()
    image=np.abs(fth.reconstruct(holo))
    mi,ma=np.percentile(image, (2,plot_max))
    ax.imshow(image,  vmin=mi, vmax=ma)
    

    add_button = widgets.Button(description="Add RH")
    output = widgets.Output()
    add_button.on_click(add_on_button_clicked)

    display(add_button, output)

    return RHdict


from scipy.ndimage.interpolation import shift
from scipy import signal
from numpy import unravel_index
def load_aligning(a,b, pad_factor=2):
    #pad them
    shape=np.array(a.shape)//2
    a=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a)))
    b=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(b)))
    padding_x=(pad_factor-1)*a.shape[1]//2
    padding_y=(pad_factor-1)*a.shape[0]//2
    pad_width=((padding_x,padding_x),(padding_y,padding_y))
    a=np.pad(a, pad_width=pad_width )
    b=np.pad(b, pad_width=pad_width )
    
    #c=signal.correlate(a,b, method="fft", mode="same")
    
    c=np.conjugate(a)*b
    c=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(c)))

    center=np.array(unravel_index(c.argmax(), c.shape))
    print(center)
    center=center/pad_factor-shape
    print(center)
    return center
    
    
def spectrogram(pos, neg, N=10, mode="+", figsize=(10,10)):
    
    image=np.zeros(pos.shape, dtype=np.complex)
    width=pos.shape[0]//N
    
    if mode=="+":
        fth_image=pos+neg
    elif mode=="-":
        fth_image=pos-neg
        
    for i in range(N):
        for j in range(N):
            image[ i* width: (i+1)*width, j* width: (j+1)*width] = fth.reconstruct(fth_image[ i* width: (i+1)*width, j* width: (j+1)*width])
            
    mi,ma=np.percentile(np.abs(image), (1,99))
    fig,ax=plt.subplots(figsize=figsize)
    ax.imshow(np.abs(image), vmin=mi, vmax=ma)
    
    return image
