################################################################################
## Miscelaneous functions used by model.py to manipulate images and mix optical constants ###
################################################################################


import numpy as np
import cmath as cma
from disc2radmc.constants import *
from astropy.io import fits
from astropy.convolution import convolve_fft
from scipy.ndimage.interpolation import shift
from scipy import interpolate

import os

# function to define vertical distribution
def rhoz_Gaussian(z, H):
    return np.exp(-(z)**2.0/(2.0*H**2.0))/(np.sqrt(2.0*np.pi)*H)

#### Functions to mix optical constants following Bruggeman's mixing rule
def effnk_bruggeman(n1,k1,n2,k2,n3,k3,f2,f3): 

    # mixing rule Bruggeman http://en.wikipedia.org/wiki/Effective_medium_approximations
    # Sum fi* (epi-ep)/(epi+2ep) = 0, but normilizing by f1
    # fi are the volume fractions normalized by f1
 
    np1=n1+k1*1j # matrix
    np2=n2+k2*1j # inclusion 1
    np3=n3+k3*1j # inclusion 2

    e1=np1**2.0  # n = sqrt(epsilon_r x mu_r) and mu_r is aprox 1
    e2=np2**2.0
    e3=np3**2.0

    # polynomial of third order
    p=np.zeros(4, dtype=complex)

    p[3]=e1*e2*e3*(1.0+f2+f3) # 0 order
    p[2]=-e1*e3*f2 -e1*e2*f3 - e2*e3 + 2*(e1*e2*f2 + e1*e2 +e1*e3*f3+e1*e3+e2*e3*f2+e2*e3*f3) # 1st order
    p[1]= -2.0*(e1*f2+e1*f3+e3*f2+e2*f3+e2+e3)+4.0*(e1+e2*f2+e3*f3)# 2nd order
    p[0]= -4.0*(1.0+f2+f3)

    roots=np.roots(p) 

    # check roots
    for i in range(len(roots)):
        effi=roots[i]
        if effi.real>0.0 and effi.imag>0.0:
            return cma.sqrt(effi)
    ### if nothing satisfies the above condition
        
    return -1.0

def effnk_mg(n1,k1,n2,k2,f2): 

    # mixing rule Maxwell-Garnett http://en.wikipedia.org/wiki/Effective_medium_approximations
    # fi are the volume fractions
 
    np1=n1+k1*1j # matrix
    np2=n2+k2*1j # inclusion 1

    e1=np1**2.0  # n = sqrt(epsilon_r x mu_r) and mu_r is aprox 1
    e2=np2**2.0
    
    return cma.sqrt(  e1*(2*f2*(e2-e1)+e2+2*e1)/(2*e1+e2-f2*(e2-e1)))
        
def Intextpol(x,y,xi):

    Nx=len(x)
    if xi<=x[0]: return y[0] # extrapol                                                                                                                                                                         
    elif xi<=x[Nx-1]: #interpol                                                                                                                                                                                                              
        for l in range(1,Nx):
            if xi<=x[l]:
                # return y[l-1]+(xi-x[l-1])*(y[l]-y[l-1])/(x[l]-x[l-1])

                return y[l-1]*(xi/x[l-1])**(np.log(y[l]/y[l-1])/np.log(x[l]/x[l-1]))


    elif xi>x[Nx-1]:    #extrapol                                                                                                                                                                                                            
        alpha=np.log(y[Nx-1]/y[Nx-2])/np.log(x[Nx-1]/x[Nx-2])
        return y[Nx-1]*(xi/x[Nx-1])**alpha


### functions to manipulate images

def convert_to_fits(path_image, path_fits, Npixf, dpc, mx=0.0, my=0.0, x0=0.0, y0=0.0, omega=0.0, fstar=-1.0, vel=False, continuum_subtraction=False, background_args=[], tag='', primary_beam=None, alpha_dust=None, new_lambda=None,verbose=False, taumap=False, fdisc=None, vr_star=0.):
    # alpha is defined as the spectral index in frequency space, and thus is positive for a typical disc and star at mm wavelengths
    
    ### load image
    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc, taumap=taumap)
    istar, jstar=star_pix(nx, omega)
    
    ## if alpha is given, then disc surface brightness and stellar flux are manipulated
    if alpha_dust is not None and new_lambda is not None:
        background=np.median([image_in_jypix[0,0,jstar-1,istar], image_in_jypix[0,0,jstar+1,istar], image_in_jypix[0,0,jstar,istar-1], image_in_jypix[0,0,jstar,istar+1]])
        Fstar=image_in_jypix[0,0,jstar,istar]-background # save stellar flux, but subtract background to not include disc.
        ### apply dust spectral index
        image_in_jypix[0,0,jstar,istar]=background
        image_in_jypix=image_in_jypix*(new_lambda/lam[0])**(-alpha_dust)
        ### apply star spectral index
        Fstar=Fstar*(new_lambda/lam[0])**(-2.0)
        image_in_jypix[0,0,istar,istar]+=Fstar

    ## if fdisc is given, then disc surface brightness and stellar flux are manipulated
    if fdisc is not None:
        ## remove the star
        background=np.median([image_in_jypix[0,0,jstar-1,istar], image_in_jypix[0,0,jstar+1,istar], image_in_jypix[0,0,jstar,istar-1], image_in_jypix[0,0,jstar,istar+1]])
        Fstar=image_in_jypix[0,0,jstar,istar]-background # save stellar flux, but subtract background to not include disc.
        ### apply dust flux
        image_in_jypix[0,0,jstar,istar]=background
        image_in_jypix=image_in_jypix*fdisc/np.sum(image_in_jypix)
        ### put back the star
        image_in_jypix[0,0,istar,istar]+=Fstar
        
    ### manipulate central flux
    if fstar>=0.0: # change stellar flux given value of fstar.
        background=np.median([image_in_jypix[:,:,jstar-1,istar], image_in_jypix[:,:,jstar+1,istar], image_in_jypix[:,:,jstar,istar-1], image_in_jypix[:,:,jstar,istar+1]])
        image_in_jypix[:, :, jstar,istar]=fstar+background
    if verbose:
        print('Fstar=', image_in_jypix[0, 0, jstar,istar])

    # PAD IMAGE
    image_in_jypix_pad=fpad_image(image_in_jypix, Npixf, Npixf, nx, ny)

    # add background sources
    if len(background_args) != 0:
        for iback in background_args:
            image_in_jypix_pad=image_in_jypix_pad + background_object(*iback)

    ### shift image if necessary
    image_in_jypix_shifted= shift_image(image_in_jypix_pad, mx, my, pixdeg_x, pixdeg_y, omega=omega)
        
    
    if primary_beam is not None:
        pbfits=fits.open(primary_beam)
        pb=pbfits[0].data[0,0,:,:] # pb image
        header_pb = pbfits[0].header # header
        # del header_pb['Origin'] # necessary due to a comment that CASA adds automatically

        # check if pixel sizes are the same
        assert abs(pixdeg_x-header_pb['CDELT2'])/pixdeg_x <0.01, ('pixel size of primary beam is %1.5e and image is %1.5e. Make sure they are the same within 1 per cent'%(pixdeg_x,header_pb['CDELT2']))

        # check if primary beam needs to be pad
        if header_pb['NAXIS1']<Npixf or header_pb['NAXIS2']<Npixf:
            pb=fpad_image(pb, Npixf, Npixf,header_pb['NAXIS1'] , header_pb['NAXIS2'])

        # multiply by primary beam and set nans to zero
        image_in_jypix_shifted=image_in_jypix_shifted*pb
        inans= np.isnan(image_in_jypix_shifted)
        image_in_jypix_shifted[inans]=0.0
        
    lam0=lam[0] # um   
    reffreq=cc/(lam0*1.0e-4) # Hz

    ### Single image
    if nf==1 and not taumap: # single image
        flux = np.sum(image_in_jypix_shifted[0,0,:,:])
        
        if verbose: print("flux [Jy] = ", flux)


    ### image cube
    elif not taumap:

        # get line rest wavelength
        # assumes line is centred so v=0 at center of the grid
        mid = nf // 2
        if nf % 2 == 0:
            lam_line=0.5 * (lam[mid - 1] + lam[mid]) # mu
        else:
            lam_line=lam[mid] # mu

        # get line rest frequency
        freq_line=cc*1.0e4/lam_line # Hz

        # contruct velocity array
        vels=(lam-lam_line)/lam_line*cc*1e-5 # km/s
        
        # apply velocity doppler shift
        vels=vels+vr_star

        # recalculate wavelength grid
        lam=lam_line*(1.+vels*1.0e5/cc)
        freq=cc*1e4/lam # Hz
        
        delta_velocity = vels[1]-vels[0] # km/s
        delta_freq= freq[1] - freq[0]    # Hz

        if continuum_subtraction: # subtract continuum assuming it varies linearly with wavelength
            if verbose: print('subtracting continuum')
            m=(image_in_jypix_shifted[0,-1,:,:]- image_in_jypix_shifted[0,0,:,:])/(lam[-1]-lam[0])
            I0=image_in_jypix_shifted[0,0,:,:]*1.
            for k in range(nf):
                Cont=I0+(lam[k]-lam[0])*m
                image_in_jypix_shifted[0,k,:,:]= image_in_jypix_shifted[0,k,:,:] - Cont 
        flux = np.sum(image_in_jypix_shifted[0,:,:,:])*delta_velocity
        if verbose: print("flux [Jy km/s] = ", flux)

    ### BUILD HEADER

    # Make FITS header information:
    header = fits.Header()
    #header['SIMPLE']='T'
    header['BITPIX']=-32
    # all the NAXIS are created automatically header['NAXIS']=2
    header['OBJECT']=tag
    header['EPOCH']=2000.0
    # header['LONPOLE']=180.0

    header['EQUINOX']=2000.0
    header['SPECSYS']='BARYCENT'
    header['VELREF']=258 # FROM ALMA cube / 1 LSR, 2 HEL, 3 OBS, +256 Radio 
    if nf==1:
        header['CTYPE3']='FREQ'
        header['CRPIX3'] = 1.0
        header['CDELT3']  = 1.0
        header['CRVAL3']= reffreq

    if not taumap:
        header['FLUX']=flux
        header['BTYPE'] = 'Intensity'
        header['BSCALE'] = 1
        header['BZERO'] = 0
        header['BUNIT'] = 'JY/PIXEL'#'erg/s/cm^2/Hz/ster'
    else:
        header['BTYPE'] = 'Tau'
        header['BSCALE'] = 1
        header['BZERO'] = 0
        header['BUNIT'] = ''#'erg/s/cm^2/Hz/ster'


    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    if Npixf%2==1: ## if odd number of pixels, central pixel coincides with center
        header['CRVAL1'] = x0
        header['CRVAL2'] = y0
        
    else:   ## if even number of pixels, there is no central pixel
        header['CRVAL1'] = x0+pixdeg_x
        header['CRVAL2'] = y0-pixdeg_y
     
    unit = 'DEG'
    multiplier = 1
    # RA
    header['CDELT1'] = -multiplier*pixdeg_x
    header['CUNIT1'] = unit
    # ...Zero point of coordinate system
    header['CRPIX1'] = (Npixf)//2+1

    # DEC
    header['CDELT2'] = multiplier*pixdeg_y
    header['CUNIT2'] = unit
    # ...Zero point of coordinate system
    header['CRPIX2'] = (Npixf)//2+1

    # FREQ
    if nf > 1:
        if vel==True:  ### not fully tested
            print(lam[int(nf//2)+1])
            # multiple frequencies - set up the header keywords to define the
            #    third axis as frequency
            header['CTYPE3'] = 'VELOCITY'
            header['CUNIT3'] = 'km/s'
            header['CRPIX3'] = int(nf//2)+1
            if nf%2==0: # even
                header['CRVAL3'] = delta_velocity/2.+vr_star
            else:
                header['CRVAL3'] = 0.0 + vr_star
            header['CDELT3']  = delta_velocity
            header['RESTFRQ'] = freq_line

        else:
            header['CTYPE3'] = 'FREQ'
            header['CUNIT3'] = 'HZ'
            header['CRPIX3'] = int(nf//2)+1
            header['CRVAL3'] = freq[nf//2]
            # Calculate the frequency step, assuming equal steps between all:
            header['CDELT3']  = delta_freq
            header['RESTFRQ'] = freq_line
    else:                # only one frequency
        header['RESTFRQ'] = reffreq
        header['CUNIT3'] = 'Hz'

    # Make a FITS file!
   
    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)
    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix', overwrite=True)



def xyarray(Np, ps_arcsec):

    xedge=np.zeros(Np+1)
    yedge=np.zeros(Np+1)

    xs=np.zeros(Np)
    ys=np.zeros(Np)

    for i in range(Np+1):

        xedge[i]=-(i-Np/2.0)*ps_arcsec#-ps_arcsec/2.0        
        yedge[i]=(i-Np/2.0)*ps_arcsec#+ps_arcsec/2.0

    for i in range(Np):
        xs[i]=-(i-Np/2.0)*ps_arcsec-ps_arcsec/2.0           
        ys[i]=(i-Np/2.0)*ps_arcsec+ps_arcsec/2.0

    return xs, ys, xedge, yedge
    
def Convolve_beam(path_image, BMAJ, BMIN, BPA, tag_out=''):

    ### BMAJ, BMIN and BPA in deg
    
    #  -----cargar fit y extraer imagen

    fit1	= fits.open(path_image) #abrir objeto cubo de datos
    
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    print(np.shape(data1))

    header1	= fit1[0].header
    ps_deg=float(header1['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    dtheta=ps_deg*np.pi/180.0 # dtheta in rad

    M=len(data1[:,0])

    N=M # dim output

    ps1= ps_mas # pixel size input in mas
    ps2= ps1 # pixel size output in mas

    dtheta2=ps2*np.pi/(3600.0*1000.0*180.0)

    d=0

    Fin1=data1[:,:]#*1e23/(dtheta**2.0) #  JY/PIXEL to ergs/s cm2 Hz sr

    x1=np.zeros(N)
    y1=np.zeros(N)
    for i in range(N):
        x1[i]=(i-M/2.0)*ps1
        y1[i]=(i-N/2.0)*ps1
  

    sigx=BMIN*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0))) 
    sigy=BMAJ*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0)))  
    theta=-BPA*np.pi/180.0   #np.pi/4.0

    # Fout1=interpol(N,M,ps1,ps2,Fin1,sigx,sigy,theta)

    Gaussimage=np.zeros((N,N))

    xs, ys, xedge, yedge = xyarray(N, ps2)
    xm, ym = np.meshgrid(xs, ys)

    Gaussimage=Gauss2d(xm,ym,0.0,0.0,sigx,sigy,theta)
    
    Fout1=convolve_fft(Fin1,Gaussimage, normalize_kernel=False, allow_huge=True)



    header1['BMIN'] = BMIN
    header1['BMAJ'] = BMAJ
    header1['BPA'] = BPA

    header1['BUNIT']='JY/BEAM'

    path_fits=path_image[:-5]+'_beamconvolved'+tag_out+'.fits'
    fits.writeto(path_fits, Fout1, header1, output_verify='fix', overwrite=True)


def Convolve_beam_cube(path_image, BMAJ, BMIN, BPA):

    ### BMAJ, BMIN and BPA in deg

    #  -----cargar fit y extraer imagen

    fit1	= fits.open(path_image) #abrir objeto cubo de datos
    
    data1 	= fit1[0].data # [0,0,:,:] # extract image matrix

    print(np.shape(data1))

    header1	= fit1[0].header
    ps_deg=float(header1['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    dtheta=ps_deg*np.pi/180.0 # dtheta in rad

    N=len(data1[0,0,0,:])
    Nf=len(data1[0,:,0,0])
    ps1= ps_mas # pixel size input in mas
    ps2= ps1 # pixel size output in mas

    dtheta2=ps2*np.pi/(3600.0*1000.0*180.0)

    d=0

    Fin1=data1[:,:]#*1e23/(dtheta**2.0) #  JY/PIXEL to ergs/s cm2 Hz sr

    x1=np.zeros(N)
    y1=np.zeros(N)
    for i in range(N):
        x1[i]=(i-N/2.0)*ps1
        y1[i]=(i-N/2.0)*ps1


   
    sigx=BMIN*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0))) 
    sigy=BMAJ*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0)))  
    theta=BPA*np.pi/180.0   #np.pi/4.0

    # Fout1=interpol(N,M,ps1,ps2,Fin1,sigx,sigy,theta)


    xs, ys, xedge, yedge = xyarray(N, ps2)
    xm, ym = np.meshgrid(xs, ys)
    Gaussimage=Gauss2d(xm,ym,0.0,0.0,sigx,sigy,theta)
    
    Fout1=np.zeros((1,Nf,N,N))
    for k in range(Nf):
        Fout1[0,k,:,:]=convolve_fft(Fin1[0,k,:,:],Gaussimage, normalize_kernel=False, allow_huge=True)


    header1['BMIN'] = BMIN
    header1['BMAJ'] = BMAJ
    header1['BPA'] = BPA

    header1['BUNIT']='Jy/beam'

    path_fits=path_image[:-5]+'_beamconvolved.fits'
    fits.writeto(path_fits, Fout1, header1, output_verify='fix', overwrite=True)



def load_image(path_image, dpc, taumap=False):

    f=open(path_image,'r')
    iformat=int(f.readline())

    if (iformat < 1) or (iformat > 4):
        assert False, "ERROR: File format of image not recognized"

    nx, ny = tuple(np.array(f.readline().split(),dtype=int))
    nf = int(f.readline()) # number of wavelengths
    sizepix_x, sizepix_y = tuple(np.array(f.readline().split(),dtype=float))

    f.close()

    f=np.loadtxt(path_image, skiprows=4) # empty line is skipped

    lam=f[:nf]
        
    image=np.zeros((1,nf,ny,nx), dtype=float)
    image[0,:,:,:] = f[nf:].reshape(nf, ny, nx)

    # Compute the flux in this image as seen at dpc (pc)    
    pixdeg_x = 180.0*(sizepix_x/(dpc*pc))/np.pi
    pixdeg_y = 180.0*(sizepix_y/(dpc*pc))/np.pi

    # Compute the conversion factor from erg/cm^2/s/Hz/ster to erg/cm^2/s/Hz/ster at dpc
    pixsurf_ster = pixdeg_x*pixdeg_y * (np.pi/180.)**2
    factor = 1e+23 * pixsurf_ster
    # And scale the image array accordingly:
    image_in_jypix = factor * image

    if taumap:
        return image, nx, ny, nf, lam, pixdeg_x, pixdeg_y
    else:
        return image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y


def star_pix(nx, omega):

    omega= omega%360.0

    if nx%2==0.0: # even number of pixels
        if omega>=0.0 and omega<=90.0:
            istar=nx//2 
            jstar=nx//2 
        elif omega>90.0 and omega<=180.0:
            istar=nx//2-1 
            jstar=nx//2 
        elif omega>180.0 and omega<=270.0:
            istar=nx//2-1 
            jstar=nx//2-1
        elif omega>270.0 and omega<360.0:
            istar=nx//2 
            jstar=nx//2 -1
    else:
        istar=nx//2
        jstar=nx//2
    return istar, jstar

def shift_image(image, mx, my, pixdeg_x, pixdeg_y, omega=0.0 ):

    if mx ==0.0 and my==0.0: return image

    mvx_pix=(mx/(pixdeg_x*3600.0))
    mvy_pix=(my/(pixdeg_y*3600.0))

    shiftVector=(0.0, 0.0, mvy_pix, -mvx_pix) # minus sign as left is positive 
    # cp star and remove it
    istar, jstar=star_pix(len(image[0,0,0,:]), omega)
    Fstar=image[0,0,jstar,istar]
    # replace star with average around it. This is important if there is a disk
    image[0,0,jstar,istar]=np.median([image[0,0,jstar-1,istar], image[0,0,jstar+1,istar], image[0,0,jstar,istar-1], image[0,0,jstar,istar+1]])
    
    # shift
    image_shifted=shift(image,shift=shiftVector, order=3)#,mode='wrap')
    # # add star in new position
    image_shifted[0,0,jstar+int(round(mvy_pix)),istar-int(round(mvx_pix))]=Fstar

    return image_shifted

def fpad_image(image_in, pad_x, pad_y, nx, ny):

    if image_in.shape[-2:] != (pad_x,pad_y):
        pad_image = np.zeros((1,1,pad_x,pad_y))
        if nx%2==0 and ny%2==0: # even number of pixels
            pad_image[0,0,
                      pad_y//2-ny//2:pad_y//2+ny//2,
                      pad_x//2-nx//2:pad_x//2+nx//2] = image_in[0,0,:,:]
        else:                  # odd number of pixels
            pad_image[0,0,
                      pad_y//2-(ny-1)//2:pad_y//2+(ny+1)//2,
                      pad_x//2-(nx-1)//2:pad_x//2+(nx+1)//2] = image_in[0,0,:,:]
        return pad_image

    else:                      # padding is not necessary as image is already the right size (potential bug if nx>pad_x)
        return image_in



def Gauss2d(xi , yi, x0,y0,sigx,sigy,theta):

    xp= (xi-x0)*np.cos(theta) + (yi-y0)*np.sin(theta)
    yp= -(xi-x0)*np.sin(theta) + (yi-y0)*np.cos(theta)

    a=1.0/(2.0*sigx**2.0)
    b=1.0/(2.0*sigy**2.0)

    return np.exp(- ( a*(xp)**2.0 + b*(yp)**2.0 ) )#/(2.0*np.pi*sigx*sigy)

def background_object(Ni, dpix, Flux, offx, offy, Rmaj, Rmin, Rpa):

    Xmax=(Ni-1)*dpix/2.
        
    xs=np.linspace(Xmax, -Xmax, Ni)
    ys=np.linspace(-Xmax, Xmax, Ni)

    Xs, Ys =np.meshgrid(xs, ys)
    rs=np.sqrt( (Xs-offx)**2. + (Ys-offy)**2. )

    F=Gauss2d(Xs , Ys, offx, offy, Rmaj, Rmin, -(Rpa+90.)*np.pi/180.)
    return F*Flux/np.sum(F)


def append_new_line(file_name, text_to_append): # from https://thispointer.com/how-to-append-text-or-lines-to-a-file-in-python/
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def delete_last_line(file_name, encoding="utf-8"): # adapted from https://stackoverflow.com/questions/1877999/delete-final-line-in-file-with-python

    with open(file_name, "r+", encoding = encoding) as file:

        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        file.seek(0, os.SEEK_END)

        # This code means the following code skips the very last character in the file -
        # i.e. in the case the last line is null we delete the last line
        # and the penultimate one
        pos = file.tell() - 1

        # Read each character in the file one at a time from the penultimate
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)

        # So long as we're not at the start of the file, delete all the characters ahead
        # of this position
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()

def delete_line_from_file(file_path, target_string):
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines that match the target string
    new_lines = [line for line in lines if target_string not in line]

    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

            
            
def get_last2d(data):
    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 2)    
    slc += [slice(None), slice(None)]
    return data[tuple(slc)]
    
def get_last3d(data):
    if data.ndim <= 3:
        return data
    slc = [0] * (data.ndim - 3)    
    slc += [slice(None), slice(None), slice(None)]
    return data[tuple(slc)]
