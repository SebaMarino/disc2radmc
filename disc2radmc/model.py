import numpy as np
import os,sys
from disc2radmc.constants import *
from disc2radmc.functions_misc import *
from astropy.io.votable import parse
home_directory = os.path.expanduser( '~' )


class simulation:
    """
    A class to run radmc3d and convert output files 
    """

    def __init__(self,  nphot=1000000, nphot_scat=1000000, nphot_spec=10000, nphot_mono=10000, scattering_mode=1, modified_random_walk=0, istar_sphere=0, tgas_eq_tdust=1, incl_lines=0, setthreads=4, rto_style=3, verbose=True):

        self.nphot=nphot
        self.nphot_scat=nphot_scat
        self.nphot_spec=nphot_spec
        self.nphot_mono=nphot_mono
        self.scattering_mode=scattering_mode
        self.modified_random_walk=modified_random_walk
        self.istar_sphere=istar_sphere
        self.tgas_eq_tdust=tgas_eq_tdust
        self.incl_lines=incl_lines
        self.setthreads=setthreads
        self.rto_style=rto_style
        self.verbose=verbose
        
        radmc_file=open('radmc3d.inp','w')
        radmc_file.write('nphot =       %1.0f \n'%self.nphot)
        radmc_file.write('nphot_scat=    %1.0f \n'%self.nphot_scat)
        radmc_file.write('nphot_spec=    %1.0f \n'%self.nphot_spec)
        radmc_file.write('nphot_mono=    %1.0f \n'%self.nphot_mono)
        radmc_file.write('scattering_mode_max = %1.0f \n'%self.scattering_mode)
        radmc_file.write('modified_random_walk = %1.0f \n'%self.modified_random_walk)
        radmc_file.write('istar_sphere= %1.0f \n'%self.istar_sphere)
        if tgas_eq_tdust==1: # if gas temperature is assumed to be same as dust 
            radmc_file.write('tgas_eq_tdust=%1.0f \n'%self.tgas_eq_tdust)            
        radmc_file.write('incl_lines = %1.0f \n'%self.incl_lines)
        radmc_file.write('lines_mode=1 \n')
        radmc_file.write('setthreads = %1.0f \n'%self.setthreads)
        radmc_file.write('rto_style = %1.0f'%self.rto_style )
        radmc_file.close()

        if not os.path.exists('./images'):
            os.makedirs('./images')

    ### methods
    # def mctherm
    # def simimage(self):
    # def simcube(self):
    # def make_sed(self):

    def mctherm(self):
        if self.verbose:
            os.system('radmc3d mctherm')
        else:
            os.system('radmc3d mctherm > mctherm.log')

    def simimage(self, dpc=1., imagename='', wavelength=880., Npix=256, dpix=0.05, inc=0., PA=0., offx=0.0, offy=0.0, X0=0., Y0=0., tag='', omega=0.0, Npixf=-1, fstar=-1.0, background_args=[], primary_beam=None, taumap=False, fields=[], fdisc=None):
        # X0, Y0, stellar position (e.g. useful if using a mosaic)
        # images: array of names for images produced at wavelengths
        # wavelgnths: wavelengths at which to produce image in um
        # fields: fields where to make images (=[0] unless observations are a mosaic)

        if Npixf==-1:
            Npixf=Npix

        sau=Npix*dpix*dpc

        if hasattr(wavelength, "__len__"):
            image_command='radmc3d image incl %1.5f  phi  %1.5f posang %1.5f  npix %1.0f  loadlambda sizeau %1.5f  secondorder'%(inc,omega, PA-90.0, Npix, sau)
            # write wavelengths into camera_wavelength_micron.inp
            Nw=len(wavelength)
            path='camera_wavelength_micron.inp'
            arch=open(path,'w')
            arch.write(str(Nw)+'\n')
            for i in range(Nw):
                arch.write('%1.8e \n'%(wavelength[i]))
            arch.close()

        else:
            image_command='radmc3d image incl %1.5f  phi  %1.5f posang %1.5f  npix %1.0f  lambda %1.5f sizeau %1.5f  secondorder'%(inc,omega, PA-90.0, Npix, wavelength, sau)

        if taumap: # compute taumap instead of image. Need to modify radmc3d.inp
            append_new_line("radmc3d.inp", 'camera_tracemode = -2')
            
        if self.verbose:
            print('image size = %1.1e au'%sau)
            print(image_command)
            os.system(image_command)
        else:
            os.system(image_command+'  > simimgaes.log')

        if taumap: # compute taumap instead of image. Need to remove last line in radmc3d.inp 
            delete_last_line("radmc3d.inp")
            
        if taumap:
            pathin ='image_'+imagename+'_'+tag+'_taumap.out'
            pathout='images/image_'+imagename+'_'+tag+'_taumap.fits'
        else:
            pathin ='image_'+imagename+'_'+tag+'.out'
            pathout='images/image_'+imagename+'_'+tag+'.fits'

        os.system('mv image.out '+pathin)

        if hasattr(offx, "__len__"): # single pointing
            for i in range(len(offx)):
                pathout='images/image_'+imagename+'.{}_'.format(fields[i])+tag+'.fits'
                convert_to_fits(pathin, pathout, Npixf, dpc, mx=offx[i], my=offy[i], x0=X0, y0=Y0, omega=omega,  fstar=fstar, background_args=background_args, tag=tag, primary_beam=primary_beam, taumap=taumap, fdisc=fdisc, verbose=self.verbose)
 
        else: # mosaic
            convert_to_fits(pathin, pathout, Npixf, dpc, mx=offx, my=offy, x0=X0, y0=Y0, omega=omega,  fstar=fstar, background_args=background_args, tag=tag, primary_beam=primary_beam, taumap=taumap, fdisc=fdisc, verbose=self.verbose)   
    
    def simcube(self, dpc=1., imagename='', mol=1, line=1, vmax=30., Nnu=20, Npix=256, dpix=0.05, inc=0., PA=0., offx=0., offy=0., X0=0., Y0=0., tag='', omega=0., Npixf=-1, fstar=-1., background_args=[], primary_beam=None, vel=False, continuum_subtraction=False, vr_star=0.0):
        # vr_star in km/s
        
        if Npixf==-1:
            Npixf=Npix

        transition='iline '+str(line)+' imolspec '+str(mol)+' widthkms %1.5f '%(vmax)+' vkms 0.0 linenlam '+str(Nnu)
        sau=Npix*dpix*dpc
        image_command='radmc3d image incl %1.5f  phi  %1.5f posang %1.5f '%(inc,omega, PA-90.0)+transition+' npix %1.0f  sizeau %1.5f  doppcatch noscat'%(Npix, sau)
        if self.verbose:
            print('image size = %1.1e au'%sau)
            print(image_command)
            os.system(image_command)
        else:
            os.system(image_command+'  > simimgaes.log')


        pathin ='image_'+imagename+'_'+tag+'.out'
        pathout='images/image_'+imagename+'_'+tag+'.fits'
        os.system('mv image.out '+pathin)
        
        convert_to_fits(pathin, pathout, Npixf, dpc, mx=offx, my=offy, x0=X0, y0=Y0, omega=omega,  fstar=fstar, continuum_subtraction=continuum_subtraction, background_args=background_args, tag=tag, primary_beam=primary_beam, verbose=self.verbose, vr_star=vr_star, vel=vel)

    def simsed(self, wavelengths=np.logspace(-1,2, 100), dpc=100., outputfile='sed.txt', inc=0., PA=0., omega=0., sizeau=0. ):

        Nw=len(wavelengths)
        path='camera_wavelength_micron.inp'
        arch=open(path,'w')
        arch.write(str(Nw)+'\n')
        for i in range(Nw):
            arch.write('%1.5e \n'%(wavelengths[i]))
        arch.close()
        
        if sizeau>0.0:
            os.system('radmc3d spectrum loadlambda incl %1.5f phi %1.5f posang %1.5f sizeau %1.5e secondorder'%(inc, omega, (PA-90.0), sizeau))
        else:
            os.system('radmc3d spectrum loadlambda incl %1.5f phi %1.5f posang %1.5f secondorder'%(inc, omega, (PA-90.0)))
        
        arch=open('spectrum.out','r')
        arch.readline()
        Nw=int(arch.readline())
        SED=np.zeros((Nw,2))
        arch.readline()
        for i in range(Nw):
            line=arch.readline()
            dat=line.split()
            SED[i,0]=float(dat[0])   # wavelength in um 
            SED[i,1]=float(dat[1])*1.0e23/dpc**2. # Flux in Jy
        arch.close()
        np.savetxt(outputfile, SED)
        return SED

    
    
class gas:
    
    """
    A class used to define the gas species, densities and velocities
    """

    def __init__(self, gas_species=None, star=None, grid=None, Masses=None, masses=None, functions_sigma=None, pars_sigma=None, h=0.05, r0=100., gamma=1.,turbulence=False, alpha_turb=None, functions_rhoz=None, mu=28. , vr=0.0, pressure_support=False, gasT=False, rc=100, Tc=20, beta=-0.5):
        assert gas_species is not None, "Gas species need to be defined"
        assert star is not None, "star needs to be defined as its mass will set the rotation speed"
        assert grid is not None, "grid object needed to define gas density distribution"
        assert functions_sigma is not None, "surface density profile needed to define gas density distribution"
        assert pars_sigma is not None, "parameters for the surface density profile needed to define gas density distribution"
        assert Masses is not None, "Total gas mass of each species not given"
        assert masses is not None, "molecular mass of each species not given"
        
        if turbulence:
            assert alpha_turb is not None, "alpha needs to be defined to set the turbulence" 
            self.alpha_turb=alpha_turb
            
        self.grid=grid
        self.gas_species=gas_species
        self.N_species=len(self.gas_species)
        self.gas_species=gas_species
        self.Masses=Masses # total gas mass of each species
        self.masses=masses # molecular mass of each species
        self.gasT=gasT # boolean of whether to use input gas temperature 
        
        if functions_rhoz==None:
            self.functions_rhoz=[]
            for ia in range(self.N_species):
                self.functions_rhoz.append(rhoz_Gaussian)
        else:
            assert len(functions_rhoz)==self.N_species, "functions_rhoz should be an array of function with a length equal to the number of species"
            self.functions_rhoz=functions_rhoz
        #################################################################
        #### create lines.inp, which indicates which species to consider
        #################################################################

        file_lines=open('lines.inp', 'w')
        file_lines.write('2 \n')
        file_lines.write(str(self.N_species)+' \n')

        for ia in range(self.N_species):
            file_lines.write(self.gas_species[ia]+'\t leiden \t 0 \t  0 \t 0 \n') # LTE, no collisional partners
        file_lines.close()



        # ### grid        
        # self.thetam, self.phim, self.rm=np.meshgrid(self.grid.th, self.grid.phi, self.grid.r, indexing='ij' ) # Theta is ordered from midplane to North pole. Theta is still the angle from the equator.
        # self.dthm, self.dphim, self.drm = np.meshgrid(self.grid.dth, self.grid.dphi, self.grid.dr, indexing='ij' )

        # self.rhom=rm*np.cos(self.thetam) 
        # self.zm=rm*np.sin(self.thetam)

        #################################################################    
        ### define the density field
        #################################################################

        self.rho_g=np.zeros((self.N_species,self.grid.Nth,self.grid.Nphi,self.grid.Nr)) # density field (only norther emisphere)

       
        # define density
    
        for ia in range(self.N_species):
            M_gas_temp= 0.0
        
            if self.grid.Nth>1: # more than one cell per emisphere
                self.rho_g[ia,:,:,:]=self.rho_3d_dens(self.grid.rhom, self.grid.phim, self.grid.zm, h, r0, gamma, self.functions_rhoz[ia], functions_sigma[ia], *pars_sigma[ia])

                # # now south emisphere is copy of nother emisphere
                # rho_d[ia,Nth-1:,:,:]=rho_d[ia,-Nth::-1,:,:]
                ### the line above works because this is how step and slicing work https://stackoverflow.com/questions/509211/understanding-slice-notation
                # a[::-1]    # all items in the array, reversed
                # a[1::-1]   # the first two items, reversed
                # a[:-3:-1]  # the last two items, reversed
                # a[-3::-1]  # everything except the last two items, reversed
            
            elif self.grid.Nth==1:# one cell

                self.rho_g[ia,:,:,:]=functions_sigma[ia](self.grid.rhom, self.grid.phim, *pars_sigma[ia])/(self.grid.dth[0]*self.grid.rhom) # rho_3d_dens(rho, 0.0, 0.0, hs, sigmaf, *args )
                # rho_d[ia,1,:,:]=rho_d[ia,0,:,:]
        
            M_gas_temp=2.*np.sum(self.rho_g[ia,:,:,:]*(self.grid.dphim*self.grid.rhom)*(self.grid.drm)*(self.grid.dthm*self.grid.rm))*au**3.0
            self.rho_g[ia,:,:,:]=self.rho_g[ia,:,:,:]*self.Masses[ia]/M_gas_temp*M_earth /self.masses[ia] # 1/cm3

        #################################################################
        ##### define velocity field
        #################################################################

        self.vel=np.zeros((3,self.grid.Nth,self.grid.Nphi,self.grid.Nr)) # gas velocity field (only norther emisphere)

        self.vel[0,:,:,:] = vr # vr, cm/s
        self.vel[1,:,:,:] = 0.0 # vtheta, cm/s
        self.vel[2,:,:,:] = np.sqrt(   G * star.Mstar*M_sun * self.grid.rhom**2/(self.grid.rm**3)/au    )  # vphi , cm/s
        self.vkep=self.vel[2,:,:,:]*1. # store the Keplerian velocity for quick access

        #################################################################
        ##### define temperature
        #################################################################

        if self.gasT:
            print('Use input gas temperature')
            self.write_gas_temperature(rc, Tc, beta)

            # remove line that tells radmc3d to use dust temperature
            delete_line_from_file('radmc3d.inp','tgas_eq_tdust=1')
            
        else:
            print('Use dust temperature')

        
        # #### define sound speed if turbulence or keplerian deviation needed (IT NEEDS TO KNOW THE SOUND SPEED)
        
        if turbulence or pressure_support: # speed in cm/s

            # WARNING: THIS WILL NOT WORK IF EMISPHERES ARE NOT MIRRORED
            if not grid.mirror: sys.exit('reading gas temperature when emispheres are not mirrored is not implemented yet') 
            
            ### load gas temperature
            if self.gasT:
                
                self.Ts=np.loadtxt('./gas_temperature.inp', skiprows=2, max_rows=self.grid.Nphi*self.grid.Nth*self.grid.Nr).reshape( (self.grid.Nphi, self.grid.Nth, self.grid.Nr))
                
            ### load dust temperature
            else:
                # if binary file exists
                try:
                    self.Ts=np.fromfile('./dust_temperature.bdat', count=self.grid.Nr*self.grid.Nphi*self.grid.Nth+4, dtype=float)[4:].reshape( (self.grid.Nphi, self.grid.Nth, self.grid.Nr))
                # if not, load normal file
                except:
                    self.Ts=np.loadtxt('./dust_temperature.inp', skiprows=3,  max_rows=self.gridmodel.Nphi*self.gridmodel.Nth*self.gridmodel.Nr).reshape( (self.grid.Nphi, self.grid.Nth, self.grid.Nr))

            # need to swap phi and Th axes
            self.Ts=np.swapaxes(self.Ts, 0,1)
            # need to reverse order of Th axis
            self.Ts=np.flip(self.Ts, axis=0) 
                
            self.cs=np.sqrt(K*self.Ts/(mu * mp)) # cm/s

            if turbulence:
                self.turbulence=np.sqrt(self.alpha_turb)*self.cs # Nth, Nphi, Nr

        if pressure_support:

            self.P=np.sum(self.rho_g*self.masses, axis=0)*self.cs**2. # cgs
            ### old version that assume dP/dR=dP/dr (spherical R = cylindrical r)
            # self.dPdr=np.gradient(self.P, self.grid.r*au, axis=2) # cgs

            ### dP/dr = dP/dR * dR/dr + dP/dtheta * dtheta/dr (derived using r,z as a function of R, theta)
            self.dPdr=np.gradient(self.P, self.grid.r*au, axis=2)*np.cos(self.grid.thetam) - np.gradient(self.P, self.grid.th, axis=0)*np.sin(self.grid.thetam)/(self.grid.rm*au) # cgs
            
            ac = G*star.Mstar*M_sun*self.grid.rhom*au**(-2)/self.grid.rm**3. 
            # add pressure deviation
            ac+= self.dPdr/np.sum(self.rho_g*self.masses, axis=0)
            # pressure term can sometimes be larger than Keplerian if gradient is too strong (e.g. exponential drop). Set ac to zero in those cases to avoid negative ac
            ac[ac<0.]=0.
    
            self.vel[2,:,:,:] = np.sqrt(ac*self.grid.rhom*au) # cm/s
            

    
                
    ###############
    ### methods ###
    ###############

    @staticmethod
    def rho_3d_dens(rho, phi, z, h, r0, gamma,  function_rhoz, function_sigma, *arguments ):
        H=h*r0*(rho/r0)**gamma # au
        return function_sigma(rho,phi, *arguments)*function_rhoz(z,H)
    

    def write_density(self):

        # Save species
        for ia in range(self.N_species):
            
            path='numberdens_'+self.gas_species[ia]+'.inp'
            file_gas=open(path,'w')
            file_gas.write('1 \n') # iformat

            if self.grid.mirror:
                file_gas.write(str((self.grid.Nr)*(self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells
            else:
                file_gas.write(str((self.grid.Nr)*(2*self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells

            for j in range(self.grid.Nphi):
                # northern emisphere
                for k in range(self.grid.Nth):
                    for i in range(self.grid.Nr):
                        file_gas.write('%1.5e \n'%(self.rho_g[ia,-(1+k),j,i]))

                if not self.grid.mirror:
                    # southern emisphere
                    for k in range(self.grid.Nth):
                        for i in range(self.grid.Nr):
                            file_gas.write('%1.5e \n'%(self.rho_g[ia,k,j,i]))
                
            file_gas.close()


    def write_velocity(self):

        path='gas_velocity.inp'
        file_velocity=open(path,'w')
        file_velocity.write('1 \n') # iformat

        if self.grid.mirror:
            file_velocity.write(str((self.grid.Nr)*(self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells
        else:
            file_velocity.write(str((self.grid.Nr)*(2*self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells

        for j in range(self.grid.Nphi):
            # northern emisphere
            for k in range(self.grid.Nth):
                for i in range(self.grid.Nr):
                    file_velocity.write(str(self.vel[0,-(1+k),j,i])+'\t '+str(self.vel[1,-(1+k),j,i])+'\t '+str(self.vel[2,-(1+k),j,i])+' \n')

            if not self.grid.mirror:
                # southern emisphere
                for k in range(self.grid.Nth):
                    for i in range(self.grid.Nr):
                        file_velocity.write(str(self.vel[0,k,j,i])+'\t '+str(self.vel[1,k,j,i])+'\t '+str(self.vel[2,k,j,i])+' \n')
                
        file_velocity.close() 

    def write_turbulence(self): 

        path='microturbulence.inp'
        file_turbulence=open(path,'w')
        file_turbulence.write('1 \n') # iformat

        if self.grid.mirror:
            file_turbulence.write(str((self.grid.Nr)*(self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells
        else:
            file_turbulence.write(str((self.grid.Nr)*(2*self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells

            
        for j in range(self.grid.Nphi):

            # northern emisphere
            for k in range(self.grid.Nth):
                for i in range(self.grid.Nr):
                    file_turbulence.write(str(self.turbulence[-(1+k),j,i])+' \n')
            if not self.grid.mirror:
                # southern emisphere
                for k in range(self.grid.Nth):
                    for i in range(self.grid.Nr):
                        file_turbulence.write(str(self.turbulence[k,j,i])+' \n')
                
        file_turbulence.close() 
        

    def gas_temperature(self, r0, T0, beta): # in spherical coordinates
        Tgas=T0*(self.grid.rm/r0)**beta
        return Tgas

    def write_gas_temperature(self, r0, T0, beta): # in spherical coordinates

        Tgas=self.gas_temperature(r0, T0, beta)

        path='gas_temperature.inp'
        
        file_gt=open(path,'w')
        file_gt.write('1 \n') # iformat
    
        if self.grid.mirror:
            file_gt.write(str((self.grid.Nr)*(self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells
        else:
            file_gt.write(str((self.grid.Nr)*(2*self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells

        for j in range(self.grid.Nphi):

            # northern emisphere
            for k in range(self.grid.Nth):
                for i in range(self.grid.Nr):
                    file_gt.write(str(Tgas[-(1+k),j,i])+' \n')
            if not self.grid.mirror:
                # southern emisphere
                for k in range(self.grid.Nth):
                    for i in range(self.grid.Nr):
                        file_gt.write(str(Tgas[k,j,i])+' \n')       
        file_gt.close() 

        
class dust:
    """
    A class used to define the dust size distribution, opacities, and density distribution d
    """
    def __init__(self, wavelength_grid, Mdust=0.1, lnk_file=None,amin=1.0, amax=1.0e4, slope=-3.5, density=3.0, N_species=1, N_per_bin=50, densities=None, mass_weights=None, tag='i', compute_opct=True, mixing_method='Bruggeman', scattering_matrix=False, porosity=0. ):
        """
        Mdust: dust mass in earth masses
        amin: minimum grain size in um
        amax: maximum grain size in um
        ....
        """

        self.lnk_file=lnk_file if lnk_file is not None else sys.exit('invalid lnk_file')
        self.wavelength_grid=wavelength_grid
        self.amin=amin
        self.amax=amax
        self.slope=slope
        self.density=density
        self.N_species=N_species
        self.N_per_bin=N_per_bin
        self.tag=tag

        ### size grid
        self.Agrid_edges=np.logspace(np.log10(self.amin), np.log10(self.amax), self.N_species+1)
        self.Agrid=np.sqrt(self.Agrid_edges[1:]*self.Agrid_edges[:-1])
        if slope!=-4:
            self.Mgrid=self.Agrid_edges[1:]**(self.slope+4.)-self.Agrid_edges[:-1]**(self.slope+4.)
        else:
            self.Mgrid=np.log(self.Agrid_edges[1:]/self.Agrid_edges[:-1])
        self.Mgrid=Mdust*self.Mgrid/np.sum(self.Mgrid)
        
        if isinstance(lnk_file, str): # if one optical constant given
            self.lnk_file_p=lnk_file
            self.density=density
            
        elif isinstance(self.lnk_file, list):
            self.densities =np.array(densities) if densities is not None else sys.exit('error in densities array')
            self.mass_weights=np.array(mass_weights) if mass_weights is not None else sys.exit('error in mass weights array')

            if len(lnk_file)==len(mass_weights) and len(lnk_file)==len(densities):
                ### compute average and save
                self.lnk_file_p='opct_'+self.tag+'.lnk'
                if compute_opct:
                    print('Compute average optical constants')
                    Opct=self.mix_opct(pathout='opct_'+self.tag+'.lnk', mixing_method=mixing_method, porosity=porosity)
            else:
                sys.exit('mass_weights or densities do not have right length')
        else:
            sys.exit('lnk_file does not have the right format (list or string)')

        path='dustopac.inp'
        file_list_opacities=open(path,'w')
        file_list_opacities.write("2               Format number of this file \n")
        file_list_opacities.write(str(self.N_species)+"              Nr of dust species \n")
        file_list_opacities.write("============================================================================ \n")
        for i in range(self.N_species):
            if scattering_matrix:
                file_list_opacities.write("10               Way in which this dust species is read \n")
            else:
                file_list_opacities.write("1               Way in which this dust species is read \n")
            file_list_opacities.write("0               0=Thermal grain \n")
            file_list_opacities.write(self.tag+"_"+str(i+1)+ " Extension of name of dustkappa_***.inp file \n")
            file_list_opacities.write("---------------------------------------------------------------------------- \n")
        file_list_opacities.close()


    ###############
    ### methods ###
    ###############
            
    ### compute opacities
    def compute_opacities(self):
        path="./"
        if not os.path.exists('./Tempkappa'):
            os.makedirs('./Tempkappa')

        for j in range(self.N_species):

            Agrid_i=np.logspace(np.log10(self.Agrid_edges[j]), np.log10(self.Agrid_edges[j+1]), self.N_per_bin)
            weights=(Agrid_i)**(self.slope+4.) # w(a) propto n(a)*m(a)*da and da propto a
            weights=weights/np.sum(weights)

            os.system('rm '+path+'Tempkappa/*')

            opctj=np.zeros((self.wavelength_grid.Nlam,3))
            for i in range(self.N_per_bin):

                os.system('rm '+path+'param.inp')
                acm=Agrid_i[i]*1.0e-4
            
                file_inp=open(path+'param.inp','w')
                file_inp.write(self.lnk_file_p[:-4]+'\n')
                e=round(np.log10(acm))
                b=acm/(10.0**e)
                file_inp.write('%1.5fd%i \n' %(b,e))
                file_inp.write('%1.5f \n' %self.density) 
                file_inp.write('1')
                file_inp.close()
                
                os.system(path+'makeopac')
                os.system('mv '+path+'dustkappa_'+self.lnk_file_p[:-4]+'.inp '+path+'Tempkappa/dustkappa_temp_'+str(i+1)+'.inp ')

                # read opacity
                opcti=np.loadtxt('./Tempkappa/dustkappa_temp_'+str(i+1)+'.inp', skiprows=2)
                opctj=opctj+weights[i]*opcti[:,1:]

            ### write opacity file
            pathout='dustkappa_'+self.tag+'_'+str(j+1)+'.inp'
            file_opacity=open(pathout,'w')
            file_opacity.write('3 \n')
            file_opacity.write(str(self.wavelength_grid.Nlam)+'\n')
            for i in range(self.wavelength_grid.Nlam):
                file_opacity.write('%f \t %f \t %f \t %f\n' %(self.wavelength_grid.lams[i],opctj[i,0],opctj[i,1],opctj[i,2]))
            file_opacity.close()

        os.system('rm ./Tempkappa/*')
        os.system('rm '+path+'param.inp')

        pathfile='dustopac.inp'
        file_list_opacities=open(pathfile,'w')
        file_list_opacities.write("2               Format number of this file \n")
        file_list_opacities.write(str(self.N_species)+"              Nr of dust species \n")
        file_list_opacities.write("============================================================================ \n")
        for i in range(self.N_species):
            file_list_opacities.write("1               Way in which this dust species is read \n")
            file_list_opacities.write("0               0=Thermal grain \n")
            file_list_opacities.write(self.tag+"_"+str(i+1)+ " Extension of name of dustkappa_***.inp file \n")
            file_list_opacities.write("---------------------------------------------------------------------------- \n")
        file_list_opacities.close()


    def mix_opct(self, pathout='opct_mix.lnk', mixing_method='Bruggeman', porosity=0.):

        # Mixing rule Bruggeman for max 3 species or Maxwell-Garnett for 2 species
        # porosity should range between 0 and 1.
        assert porosity<1. and porosity>=0., "Porosity should range between [0,1)"
        assert mixing_method=='Bruggeman' or  mixing_method=='MG', "Mixing method different from Bruggeman or MG (Maxwell-Garnett)"
        
        N_opct=len(self.lnk_file)
       

        self.volumes=self.mass_weights/self.densities
       
        self.volume_weights=self.volumes/np.sum(self.volumes) # volume fractions that sum 1
      
        self.voli=self.volume_weights[:]/self.volume_weights[0] # normalized to the volume weight of the first species as required in Bruggeman rule's implementation

        self.density_solid=np.sum(self.mass_weights)/np.sum(self.volumes) # solid density before incorporating porosity

        self.porosity=porosity

        self.density=self.density_solid*(1.-self.porosity) # solid density before incorporating porosity
        
        print("final density = %1.1f g/cm3"%self.density)


        O1=np.loadtxt(self.lnk_file[0] ) # array for effective refractive index (n+jk)
        
        if N_opct>1:
            O2=np.loadtxt(self.lnk_file[1] )
        if N_opct==3:
            O3=np.loadtxt(self.lnk_file[2] )
            
        Opct1=np.zeros((self.wavelength_grid.Nlam,3))
        Opct1[:,0]=self.wavelength_grid.lams

        for i in range(self.wavelength_grid.Nlam):

            n1=Intextpol(O1[:,0],O1[:,1],Opct1[i,0])
            k1=Intextpol(O1[:,0],O1[:,2],Opct1[i,0])

            if N_opct>1:
                n2=Intextpol(O2[:,0],O2[:,1],Opct1[i,0])
                k2=Intextpol(O2[:,0],O2[:,2],Opct1[i,0])

            if N_opct==3:
                n3=Intextpol(O3[:,0],O3[:,1],Opct1[i,0])
                k3=Intextpol(O3[:,0],O3[:,2],Opct1[i,0])
                eff=effnk_bruggeman(n1,k1,n2,k2,n3,k3,self.voli[1],self.voli[2])

            elif N_opct==2 and mixing_method=='Bruggeman':
                eff=effnk_bruggeman(n1,k1,n2,k2,0.,0.,self.voli[1],0.)

            elif N_opct==2 and mixing_method=='MG':
                eff=effnk_mg(n1,k1,n2,k2,self.volume_weights[1])
                
            else: # one species
                eff=n1+k1*1j

            Opct1[i,1]=eff.real
            Opct1[i,2]=eff.imag

        # now we add porosity using Maxwell Garnet Equation
        if self.porosity>0: 
            vol_vac=self.porosity
            Opctf=np.zeros((self.wavelength_grid.Nlam,3))
            Opctf[:,0]=self.wavelength_grid.lams

            for i in range(self.wavelength_grid.Nlam):

                # refractive index of vacuum
                nv=1.000001 # Vosgchinnikov+2005 https://www.aanda.org/articles/aa/abs/2005/02/aa3679/aa3679.html
                kv=0.       # Vosgchinnikov+2005 https://www.aanda.org/articles/aa/abs/2005/02/aa3679/aa3679.html

                if mixing_method=='MG':
                    eff=effnk_mg(Opct1[i,1],Opct1[i,2],nv,kv,vol_vac)
                else:
                    eff=effnk_bruggeman(Opct1[i,1],Opct1[i,2],nv,kv,0.,0.,vol_vac/(1-vol_vac),0.)

                Opctf[i,1]=eff.real
                Opctf[i,2]=eff.imag
                
        else:
            Opctf=Opct1

        np.savetxt(pathout,Opctf)

    

    def dust_densities(self, grid=None, function_sigma=None, par_sigma=None, h=0.05, r0=100., gamma=1., a0=1., beta=0., functions_rhoz=None, size_segregation=False):

        if functions_rhoz==None:
            self.functions_rhoz=[]
            for ia in range(self.N_species):
                self.functions_rhoz.append(rhoz_Gaussian)
        else:
            assert len(functions_rhoz)==self.N_species, "functions_rhoz should be an array of function with a length equal to the number of species"
            self.functions_rhoz=functions_rhoz
        
        """
        funtion_sigma: function that defines the surface density. The first two arguments are two nd numpy arrays for rho and phi
        """
        
        assert grid is not None, "grid object needed to define dust density distribution"
        assert function_sigma is not None, "surface density profile needed to define dust density distribution"

        ## strange cases ntheta=1, mirror, etc
        self.grid=grid
        self.rho_d=np.zeros((self.N_species,self.grid.Nth,self.grid.Nphi,self.grid.Nr)) # density field (only norther emisphere)

        # thetam, phim, rm=np.meshgrid(self.grid.th, self.grid.phi, self.grid.r, indexing='ij' ) # so it goes from Northpole to equator. theta is still the angle from the equator.
        # dthm, dphim, drm = np.meshgrid(self.grid.dth, self.grid.dphi, self.grid.dr, indexing='ij' )
        # rhom=rm*np.cos(thetam) 
        # zm=rm*np.sin(thetam)

        
        for ia in range(self.N_species):
            M_dust_temp= 0.0

            if size_segregation:
                # nother emisphere
                if self.grid.Nth>1: # more than one cell per emisphere
                    self.rho_d[ia,:,:,:]=self.rho_3d_dens_seg(self.grid.rhom, self.grid.phim, self.grid.zm, h, r0, gamma, self.Agrid[ia], a0, beta, self.functions_rhoz[ia], function_sigma, *par_sigma)

                elif self.grid.Nth==1:# one cell
                    self.rho_d[ia,:,:,:]=function_sigma(self.grid.rhom, self.grid.phim, self.Agrid[ia], *par_sigma)/(self.grid.dth[0]*self.grid.rhom) 

            else:
                # nother emisphere
                if self.grid.Nth>1: # more than one cell per emisphere
                    self.rho_d[ia,:,:,:]=self.rho_3d_dens(self.grid.rhom, self.grid.phim, self.grid.zm, h, r0, gamma, self.functions_rhoz[ia], function_sigma, *par_sigma)

                elif self.grid.Nth==1:# one cell
                    self.rho_d[ia,:,:,:]=function_sigma(self.grid.rhom, self.grid.phim, *par_sigma)/(self.grid.dth[0]*self.grid.rhom) 

                
            M_dust_temp=2.*np.sum(self.rho_d[ia,:,:,:]*(self.grid.dphim*self.grid.rhom)*(self.grid.drm)*(self.grid.dthm*self.grid.rm))*au**3.0
            self.rho_d[ia,:,:,:]=self.rho_d[ia,:,:,:]*self.Mgrid[ia]/M_dust_temp*M_earth


    ## dust density from X,Y,Z positions (same density for all species in current implementation)
    def dust_densities_Nbody(self, grid=None, positions=[], fmt='XYZ'):

        # positions: [au or rad],  array with position of particles with shape (Nparticles, Ndim) 
        
        
        assert grid is not None, "grid object needed to define dust density distribution"
        assert positions!=[], "empty array with positions"

        self.grid=grid
        self.rho_d=np.zeros((self.N_species,self.grid.Nth,self.grid.Nphi,self.grid.Nr)) # density field (only norther emisphere)

        # thetam, phim, rm=np.meshgrid(self.grid.th, self.grid.phi, self.grid.r, indexing='ij' ) # so it goes from Northpole to equator. theta is still the angle from the equator.
        # dthm, dphim, drm = np.meshgrid(self.grid.dth, self.grid.dphi, self.grid.dr, indexing='ij' )
        # rhom=rm*np.cos(thetam) 
        # zm=rm*np.sin(thetam)
        
        if fmt=='XYZ':
            # array with x,y,z positions in au
            print('converting xyz positions to theta, phi, r')
            rs=np.sqrt(positions[:,0]**2+positions[:,1]**2+positions[:,2]**2) # radius in sphe
            rhos=np.sqrt(positions[:,0]**2+positions[:,1]**2) # radius in polar coordinates
            phis=np.arctan2(positions[:,1], positions[:,0]) # from -pi to pi
            phis[phis<0.]=phis[phis<0.]+2*np.pi # from 0 to 2pi
            thetas=np.arctan2(positions[:,2], rhos)
            pos=np.array( [thetas,phis,rs] ).T
            
        elif fmt=='THETAPHIR':
            print('positions in theta, phi, r format')
            pos=positions
            
        elif fmt=='RPHITHETA':
            print('converting r,  phi, theta to theta, phi, r')
            pos=np.array([positions[:,2],positions[:,1],positions[:,0]]).T
        else:
            sys.exit('Not a valid format ')
            
        density, edges=np.histogramdd( pos, bins=(self.grid.thedge,self.grid.phiedge,self.grid.redge), density=True)

        
        for ia in range(self.N_species):
            M_dust_temp= 0.0

            # nother emisphere
            if self.grid.Nth>1: # more than one cell per emisphere
                self.rho_d[ia,:,:,:]=density

            elif self.grid.Nth==1:# one cell
                self.rho_d[ia,:,:,:]=density

            M_dust_temp=2.*np.sum(self.rho_d[ia,:,:,:]*(self.grid.dphim*self.grid.rhom)*(self.grid.drm)*(self.grid.dthm*self.grid.rm))*au**3.0
            self.rho_d[ia,:,:,:]=self.rho_d[ia,:,:,:]*self.Mgrid[ia]/M_dust_temp*M_earth
            


    ###############
    ### methods ###
    ###############
    @staticmethod
    def rho_3d_dens(rho, phi, z, h, r0, gamma,  function_rhoz, function_sigma, *arguments ):
        H=h*r0*(rho/r0)**gamma # au
        return function_sigma(rho,phi, *arguments)*function_rhoz(z,H)

    @staticmethod
    def rho_3d_dens_seg(rho, phi, z, h, r0, gamma, a, a0=1., beta=0., function_rhoz=rhoz_Gaussian, function_sigma=None, *arguments ):
        """
        a0   :  reference grain size in um
        a    :  grain size in um
        beta : exponent of vertical distribution segregation 
        """
        
        H=h*r0*(rho/r0)**gamma * (a/a0)**beta # au
        return function_sigma(rho,phi, a, *arguments)*function_rhoz(z,H)
            
        
    def write_density(self):

        # Save 
        path='dust_density.inp'
        file_dust=open(path,'w')
    
        file_dust.write('1 \n') # iformat
        if self.grid.mirror:
            file_dust.write(str((self.grid.Nr)*(self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells
        else:
            file_dust.write(str((self.grid.Nr)*(2*self.grid.Nth)*(self.grid.Nphi))+' \n') # iformat n cells

        file_dust.write(str(self.N_species)+' \n') # n species

        for ia in range(self.N_species):
            for j in range(self.grid.Nphi):

                # northern emisphere
                for k in range(self.grid.Nth):
                    for i in range(self.grid.Nr):
                        file_dust.write(str(self.rho_d[ia,-(1+k),j,i])+' \n')

                if not self.grid.mirror:
                    # southern emisphere
                    for k in range(self.grid.Nth):
                        for i in range(self.grid.Nr):
                            file_dust.write(str(self.rho_d[ia,k,j,i])+' \n')
        file_dust.close()
        




        

class star:
    """
    A class to define the star and companions
    """
    def __init__(self, lam_grid,
                 dir_stellar_templates=home_directory+'/Astronomy/Stellar_templates/BT-Settl/bt-settl/',
                 Tstar=None,
                 Rstar=None,
                 Mstar=None,
                 g=4.
                 # companion=False
                 # separation=0.0,
                 # inc=0.0,
                 # plt=False, T_plt=0.0, R_plt=0.0, M_plt=0.0
                 ):

        default_Tstar=5750.
        default_Rstar=1.0
        default_Mstar=1.0
        
        if Tstar is not None:
            if (Tstar>=2000. and Tstar<=10000.):
                self.Tstar=Tstar
            elif Tstar<0.0:
                self.Tstar=Tstar
        else: self.Tstar= default_Tstar
           
        if Rstar is not None:
            self.Rstar=Rstar if Rstar>0. else default_Rstar
        else:
            self.Rstar=default_Rstar
        if Mstar is not None:
            self.Mstar=Mstar if Mstar>0. else default_Mstar
        else:
            self.Mstar=default_Mstar
        self.g=g
        
        self.Nlam=lam_grid.Nlam
        self.lams=lam_grid.lams
        self.dlams=lam_grid.dlams
        self.model_directory=dir_stellar_templates
        #########################
        ###### load spectra
        #########################
        # Temperature resolution of NextGen grid

        if self.Tstar>0.0:

            if self.Tstar<7000.: 
                dT=100.
            else:
                dT=200.
            
            if self.Tstar%dT==0.: # no need to interpolate
                spectrum=self.get_spectrum(self.Tstar, self.g)
            else:
              
                T1=dT*(self.Tstar//dT)
                T2=T1+dT

                spectrum1=self.get_spectrum(T1, self.g)
                spectrum2=self.get_spectrum(T2, self.g)

                w1=abs(T1-self.Tstar)
                w2=abs(T2-self.Tstar)
                spectrum=(spectrum1*w1+spectrum2*w2)/(w1+w2)

                
            ### compute flux at 1pc
            # spectrum is in units of erg/cm2/s/A
            # we want to convert Flambda to Fnu so we need to multiply by lambda**2/c
            spectrum_fnu=spectrum*(self.lams*1.0e4)**2.0/cc_a # now in units of erg/cm2/s/Hz

            ## flux at 1pc
            self.flux_1pc=spectrum_fnu*(self.Rstar*R_sun/pc)**2.

        


        
    def get_spectrum(self, T, g=4.):
        """
        returns model spectrum in units of erg/cm2/s/A 
        """
        if T>=2600.:
            path=self.model_directory+'lte%03i-%1.1f-0.0a+0.0.BT-NextGen.7.dat.txt'%(T//100,g)
        else:
            path=self.model_directory+'lte%03i-%1.1f-0.0.BT-Settl.7.dat.txt'%(T//100,g)

        data=np.loadtxt(path)
        
        Nlam_model=data.shape[0]
    
        # down sample spectrum
        spectrum_lowr=np.zeros(self.Nlam)
        for i in range(self.Nlam):
            if self.lams[i]<1.0e3:
                mask=(data[:,0]/1.0e4>(self.lams[i]-self.dlams[i]/2.)) & (data[:,0]/1.0e4<(self.lams[i]+self.dlams[i]/2.)) # data in Armstrong and lams in um
                spectrum_lowr[i]=np.mean(data[mask,1])
            else: # Rayleigh Jeans extrapolation
                spectrum_lowr[i]=spectrum_lowr[i-1]*(self.lams[i]/self.lams[i-1])**(-4.)
        return spectrum_lowr
        
       
        
    def save(self):

        path='stars.inp'
        file_star=open(path,'w')
        file_star.write('2 \n')

        # if plt:
        # file_star.write('2  '+str(Nw)+'\n')
        # else:

        file_star.write('1  '+str(self.Nlam)+'\n')
        file_star.write(str(self.Rstar*R_sun)+'\t'+str(self.Mstar*M_sun)+'\t 0.0  0.0  0.0  \n')
        # if plt:
        #     file_star.write(str(R_plt)+'\t'+str(M_plt)+'\t'+'0.0 0.0 0.0   \n')

        ### wavelengths
        for i in range(self.Nlam):
            file_star.write(str(self.lams[i])+'\n')
        ### star
        if self.Tstar>0.0:
            for i in range(self.Nlam):
                file_star.write(str(self.flux_1pc[i])+'\n')
        else:
            file_star.write(str(self.Tstar)+'\n')
        # ### planet
        # if plt:
        #     file_star.write(str(-T_plt)+'\n')

        file_star.close()

class wavelength_grid:
    """
    A class used to define the wavelength grid (in um)
    """

    def __init__(self,lammin=None, lammax=None, Nlam=None):

        if lammin is not None:
            self.lammin=lammin if lammin>0. else 0.09 # um
        else: self.lammin=0.09
        if lammax is not None:
            self.lammax=lammax if lammax>self.lammin else 1.0e5 # um
        else: self.lammax=1.0e5
        if Nlam is not None:
            self.Nlam=int(Nlam) if Nlam>0 else 150
        else: self.Nlam=150

        self.lams=np.logspace(np.log10(self.lammin), np.log10(self.lammax), self.Nlam)
        P=(self.lams[1]-self.lams[0])/self.lams[0]
        self.dlams=P*self.lams

        self.nu=cc*1.0e4/self.lams
        self.dnu=self.dlams*(cc*1.0e4/self.lams**2) # Hz
        
    def save(self):
        # ----- write wavelength_micron.inp

        path='wavelength_micron.inp'
        file_lams=open(path,'w')
        file_lams.write(str(self.Nlam)+'\n')
        for i in range(self.Nlam):
            file_lams.write(str(self.lams[i])+'\n')
        file_lams.close()



    
    
class physical_grid:
    """
    A class used to define and store grid parameters

    Theta is defined from midplane (theta=0) increasing towards the N pole (theta=pi/2), but in radmc3d theta=0 is the N pole and theta=90 is midplane.

    """
    

    
    def __init__(self, Nr=None, Nphi=None, Nth=None, rmin = None, rmax=None, thmin=None, thmax=None, logr=False, logtheta=False,  axisym=False, mirror=True, save=True, load=False):


        default_Nr=100
        default_Nphi=100
        default_Nth=20

        default_rmin=1.0
        default_rmax=300.
        default_thmin=0.03/10.
        default_thmax=0.1*3

        self.axisym=axisym
        self.logtheta=logtheta
        self.logr=logr
        self.mirror=mirror
        
        if rmin is not None:
            self.rmin=rmin if rmin>0. else default_rmin
        else:  self.rmin=default_rmin
            
        if rmax is not None:
            self.rmax=rmax if rmax>self.rmin else default_rmax
        else: self.rmax=default_rmax

        if thmin is not None:
            self.thmin=thmin if thmin>0. and thmin<np.pi/2.  else default_thmin
        else: self.thmin=default_thmin

        if thmax is not None:
            self.thmax=thmax if thmax<=np.pi/2 and (thmax>self.thmin or not self.logtheta) else default_thmax
        else: self.thmax=default_thmax
        
        if Nr is not None:
            self.Nr=int(Nr) if Nr>0 else default_Nr
        else: self.Nr=default_Nr
        
        if Nth is not None:
            self.Nth=int(Nth) if Nth>0 else default_Nth
        else: self.Nth=default_Nth
        
        if self.axisym:
            self.Nphi=1
        else:
            if Nphi is not None:
                self.Nphi=int(Nphi) if Nphi>0 else default_Nphi
                
            else: self.Nphi=default_Nphi
            if self.Nphi==1: self.axisym=True # enforce axisym if Nphi is set to 1

        ### R
        if self.logr: # log sampling
            self.redge=np.logspace(np.log10(self.rmin), np.log10(self.rmax), self.Nr+1)
        else:
            self.redge=np.linspace(self.rmin, self.rmax, self.Nr+1)
        self.dr=self.redge[1:]-self.redge[:-1]
        self.r=(self.redge[1:]+self.redge[:-1])/2.


        ### Theta (measured from midplane)
        if self.logtheta and self.Nth>2: # log sampling

            self.thedge=np.zeros(self.Nth+1)
            self.thedge[0]=0.0
            self.thedge[1:]=np.logspace(np.log10(self.thmin), np.log10(self.thmax), self.Nth)
        else:
            self.thedge=np.linspace(0., self.thmax, self.Nth+1)
            self.thmin=self.thedge[1]

        self.dth=self.thedge[1:]-self.thedge[:-1]
        self.th=(self.thedge[1:]+self.thedge[:-1])/2
    
    
        ### Phi

        # ### linear sampling in phi
        self.phiedge=np.linspace(0.,2.0*np.pi, self.Nphi+1)
        self.phi=(self.phiedge[1:]+self.phiedge[:-1])/2.
        self.dphi=self.phiedge[1:]-self.phiedge[:-1]


        self.Ncells=self.Nr*self.Nphi*self.Nth
        if self.mirror==False: self.Ncells=self.Ncells*2


        self.thetam, self.phim, self.rm=np.meshgrid(self.th, self.phi, self.r, indexing='ij' ) # Theta is ordered from midplane to North pole. Theta is still the angle from the equator.
        self.dthm, self.dphim, self.drm = np.meshgrid(self.dth, self.dphi, self.dr, indexing='ij' )

        self.rhom=self.rm*np.cos(self.thetam) 
        self.zm=self.rm*np.sin(self.thetam)
        
    def save(self):
    
        path='amr_grid.inp' #'amr_grid.inp'

        gridfile=open(path,'w')
        gridfile.write('1 \n') # iformat: the format number, at present 1
        gridfile.write('0 \n') # Grid style (regular = 0)
        gridfile.write('101 \n') # coordsystem: If 100<=coordsystem<200 the coordinate system is spherical
        gridfile.write('0 \n') # gridinfo

        if self.axisym and self.Nphi==1:
            gridfile.write('1 \t 1 \t 0 \n') # incl x, incl y, incl z
        else:
            gridfile.write('1 \t 1 \t 1 \n') # incl x, incl y, incl z
            
        if self.mirror:
            gridfile.write(str(self.Nr)+ '\t'+ str(self.Nth)+'\t'+ str(self.Nphi)+'\n') 
        else:
            gridfile.write(str(self.Nr)+ '\t'+ str(self.Nth*2)+'\t'+ str(self.Nphi)+'\n') 

        for i in range(self.Nr+1):
            gridfile.write(str(self.redge[i]*au)+'\t')
        gridfile.write('\n')

        for i in range(self.Nth+1):
            gridfile.write(str(np.pi/2.0-self.thedge[self.Nth-i])+'\t')  # from northpole to equator
        if not self.mirror:
            for i in range(1,self.Nth+1):
                gridfile.write(str(np.pi/2.0+self.thedge[i])+'\t')       # from 0 to -pi/2
        gridfile.write('\n')

        for i in range(self.Nphi+1):
            gridfile.write(str(self.phiedge[i])+'\t')
        gridfile.write('\n')
            
        gridfile.close()


    def load(self):
        print('in progress')
            




