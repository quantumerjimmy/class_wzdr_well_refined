import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from classy import Class

rc('font',**{'family':'serif','serif':['Times']}, **{'weight':'heavy'})
rc('text', usetex=True)


font = {'size': 16, 'family': 'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)

#%%

lTT,DlTT_mean,DlTT_error_minus,DlTT_error_plus,DlTT_bestfit= np.loadtxt("error_Planck/Planck2018_errorTT.txt",unpack=True)
lEE,DlEE_mean,DlEE_error_minus,DlEE_error_plus,DlEE_bestfit= np.loadtxt("error_Planck/Planck2018_errorEE.txt",unpack=True)
lTE,DlTE_mean,DlTE_error_minus,DlTE_error_plus,DlTE_bestfit= np.loadtxt("error_Planck/Planck2018_errorTE.txt",unpack=True)

#kk = np.logspace(-3,0,1000) # k in h/Mpc
kk = np.logspace(-3,0,1000) # k in 1/Mpc

Pk_lcdm = [] # P(k) in (Mpc/h)**3
Pk_wzdr = [] # P(k) in (Mpc/h)**3
Pk_idm_wzdr = [] # P(k) in (Mpc/h)**3




############ FIRST MODEL: BASE LCDM ###########################################

M = Class()
M.set({'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'l_max_scalars':2600,
                   'format':'camb',
                   'recombination' :'HyRec',
                   'H0':67.32,
                   'omega_b':0.022383, 
                   'omega_cdm':0.12011,
                   'ln10^{10}A_s':3.0448,
                   'n_s': 0.96605,
                   'tau_reio':0.0543,
                   'P_k_max_1/Mpc':10.0,
                   'N_ur':2.0328,
                   'N_ncdm':1,
                   'm_ncdm':0.06,
                   'T_ncdm': 0.71611,
                   'input_verbose': 1,
                   'background_verbose' :1,
                   'thermodynamics_verbose' :1,
                   'perturbations_verbose': 1,
                   'transfer_verbose': 1,
                   'primordial_verbose': 1,
                   'harmonic_verbose': 1,
                   'fourier_verbose': 1,
                   'lensing_verbose' :1,
                   'output_verbose': 1
                   })
M.compute()

clM = M.lensed_cl(2600)
ll_lcdm = clM['ell'][2:]
clTT_lcdm = clM['tt'][2:]
clEE_lcdm = clM['ee'][2:]
clpp_lcdm = clM['pp'][2:]

h = M.h() # get reduced Hubble for conversions to 1/Mpc
# get P(k) at redhsift z=0
for k in kk:
#    Pk_lcdm.append(M.pk(k*h,0.)*h**3) # function .pk(k,z)
    Pk_lcdm.append(M.pk(k,0.)) # function .pk(k,z)


M.struct_cleanup()
M.empty()

fTT_lcdm = interp1d(ll_lcdm,ll_lcdm*(ll_lcdm+1.0)*pow(2.7255*1.e6,2)*clTT_lcdm/(2.0*np.pi))
fEE_lcdm = interp1d(ll_lcdm,ll_lcdm*(ll_lcdm+1.0)*pow(2.7255*1.e6,2)*clEE_lcdm/(2.0*np.pi))
fpp_lcdm = interp1d(ll_lcdm,ll_lcdm*(ll_lcdm+1.0)*pow(2.7255*1.e6,2)*clpp_lcdm/(2.0*np.pi))


fpk_lcdm = interp1d(kk,Pk_lcdm)

############ SECOND MODEL: WZDR ###############################################

M = Class()
M.set({'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'l_max_scalars':2600,
                   'format':'camb',
                   'recombination' :'HyRec',
                   '100*theta_s':1.042351,
                   'omega_b':0.02263473, 
                   'omega_cdm':0.1242812,
                   'ln10^{10}A_s':3.055681,
                   'n_s': 0.972601,
                   'tau_reio':0.05993115,
                   'P_k_max_1/Mpc':10.0,
                   'N_ur':2.0328,
                   'N_ncdm':1,
                   'm_ncdm':0.06,
                   'T_ncdm': 0.71611,
                   'input_verbose': 1,
                   'background_verbose' :1,
                   'thermodynamics_verbose' :1,
                   'perturbations_verbose': 1,
                   'transfer_verbose': 1,
                   'primordial_verbose': 1,
                   'harmonic_verbose': 1,
                   'fourier_verbose': 1,
                   'lensing_verbose' :1,
                   'output_verbose': 1
                   })
M.set({ 
     'N_wzdr': 0.3,
     'zt_wzdr' : 10**(4.298095e+00),
     'rg_wzdr':1.14285714286,
     'g2_wzdr': 2,
     'spinstat_wzdr':-1,
     'wzdr_no_bbn':'no',
     'use_wzdr_PSD':'yes',
     'wzdr_integration_method': 1,
     'tol_wzdr':1e-15,
     'qsize_wzdr':10000,
     'qmax_wzdr':200.,
     'use_wzdr_PSD':'yes'
    })

M.compute()
clM = M.lensed_cl(2600)
ll_wzdr = clM['ell'][2:]
clTT_wzdr = clM['tt'][2:]
clEE_wzdr = clM['ee'][2:]
clpp_wzdr = clM['pp'][2:]

  
h = M.h() # get reduced Hubble for conversions to 1/Mpc
# get P(k) at redhsift z=0
for k in kk:
#    Pk_wzdr.append(M.pk(k*h,0.)*h**3) # function .pk(k,z)
    Pk_wzdr.append(M.pk(k,0.)) # function .pk(k,z)

    
M.struct_cleanup()
M.empty()

fTT_wzdr = interp1d(ll_wzdr,ll_wzdr*(ll_wzdr+1.0)*pow(2.7255*1.e6,2)*clTT_wzdr/(2.0*np.pi))
fEE_wzdr = interp1d(ll_wzdr,ll_wzdr*(ll_wzdr+1.0)*pow(2.7255*1.e6,2)*clEE_wzdr/(2.0*np.pi))
fpp_wzdr = interp1d(ll_wzdr,ll_wzdr*(ll_wzdr+1.0)*pow(2.7255*1.e6,2)*clpp_wzdr/(2.0*np.pi))
fpk_wzdr = interp1d(kk,Pk_wzdr)

############ THIRD MODEL: IDM-WZDR ############################################

M = Class()
M.set({'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'l_max_scalars':2600,
                   'format':'camb',
                   'recombination' :'HyRec',
                   '100*theta_s':1.042351,
                   'omega_b':0.02263473, 
                   'omega_cdm':0.1242812,
                   'ln10^{10}A_s':3.055681,
                   'n_s': 0.972601,
                   'tau_reio':0.05993115,
                   'P_k_max_1/Mpc':10.0,
                   'N_ur':2.0328,
                   'N_ncdm':1,
                   'm_ncdm':0.06,
                   'T_ncdm': 0.71611,
                   'input_verbose': 1,
                   'background_verbose' :1,
                   'thermodynamics_verbose' :1,
                   'perturbations_verbose': 1,
                   'transfer_verbose': 1,
                   'primordial_verbose': 1,
                   'harmonic_verbose': 1,
                   'fourier_verbose': 1,
                   'lensing_verbose' :1,
                   'output_verbose': 1
                   })
M.set({ #These values of N_wzdr and zt are the mean values taken from the Planck18TTTEEE+BAO+Pantheon analysis
     'N_wzdr': 0.3,
     'zt_wzdr' : 10**(4.298095e+00),
     'f_idm_wzdr':1.0,  #I think this option is not compatible with the "Omega_m as input" option
     'Gamma_0_wzdr':1.0e-7, #units of Mpc^{-1}, chosen because it should lead to a significant suppression in Pk (but is ruled out by CMB)
     'm_idm_wzdr': 1.0e11,
     'nindex_idm_wzdr': 0,
     'rg_wzdr':1.14285714286,
     'g2_wzdr': 2,
     'spinstat_wzdr':-1,
     'wzdr_no_bbn':'no',
     'use_wzdr_PSD':'yes',
     'wzdr_integration_method': 1,
     'tol_wzdr':1e-15,
     'qsize_wzdr':10000,
     'qmax_wzdr':200.,
     'use_wzdr_PSD':'yes'
    })

M.compute()
clM = M.lensed_cl(2600)
ll_idm_wzdr = clM['ell'][2:]
clTT_idm_wzdr = clM['tt'][2:]
clEE_idm_wzdr = clM['ee'][2:]
clpp_idm_wzdr = clM['pp'][2:]

h = M.h() # get reduced Hubble for conversions to 1/Mpc
# get P(k) at redhsift z=0
for k in kk:
#    Pk_idm_wzdr.append(M.pk(k*h,0.)*h**3) # function .pk(k,z)
    Pk_idm_wzdr.append(M.pk(k,0.)) # function .pk(k,z)


M.struct_cleanup()
M.empty()

fTT_idm_wzdr = interp1d(ll_idm_wzdr,ll_idm_wzdr*(ll_idm_wzdr+1.0)*pow(2.7255*1.e6,2)*clTT_idm_wzdr/(2.0*np.pi))
fEE_idm_wzdr = interp1d(ll_idm_wzdr,ll_idm_wzdr*(ll_idm_wzdr+1.0)*pow(2.7255*1.e6,2)*clEE_idm_wzdr/(2.0*np.pi))
fpp_idm_wzdr = interp1d(ll_idm_wzdr,ll_idm_wzdr*(ll_idm_wzdr+1.0)*pow(2.7255*1.e6,2)*clpp_idm_wzdr/(2.0*np.pi))
fpk_idm_wzdr = interp1d(kk,Pk_idm_wzdr)

#%% PLOT P(K) residuals

fig, axes = plt.subplots()

axes.set_xscale('log')
axes.set_xlim(kk[0],kk[-1])
axes.set_ylim(-100.*0.18,100.*0.015)

#axes.set_xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$', fontsize=20)
axes.set_xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$', fontsize=20)
axes.set_ylabel(r'relative difference $P(k)$ in \%', fontsize=20)


axes.plot(kk,100.*(fpk_wzdr(kk)/fpk_lcdm(kk)-1.0),'red',linestyle='dashed',label=r'$\mathrm{WZDR}$')
axes.plot(kk,100.*(fpk_idm_wzdr(kk)/fpk_lcdm(kk)-1.0),'green',label=r'$\mathrm{IDM}-\mathrm{WZDR}$')

plt.legend(fontsize =18,loc='lower left',borderaxespad=2.)

k_range_sigma8 = np.linspace(0.1,0.9,1000) 
plt.fill_between(k_range_sigma8, -100,100, color='paleturquoise')

plt.axhline(y=0.0, color='blacK', linestyle='-')

#plt.text(0.02,0.23,r'$z=0$',fontsize=20)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)

plt.show()
plt.clf()

#%% PLOT CMB TT residuals

fig, axes = plt.subplots()
plt.ylim([-100.*0.05,100.*0.05])
plt.xlim([2,2600])

plt.xlabel(r'$\ell$',fontsize=20)
plt.ylabel(r'relative difference $C_{\ell}^{\mathrm{TT}}$ in \%',fontsize=20)

plt.plot(ll_lcdm,100.*(fTT_wzdr(ll_lcdm)/fTT_lcdm(ll_lcdm)-1),'red',linestyle='dashed',label=r'$\mathrm{WZDR}$')
plt.plot(ll_lcdm,100.*(fTT_idm_wzdr(ll_lcdm)/fTT_lcdm(ll_lcdm)-1),'green',label=r'$\mathrm{IDM}-\mathrm{WZDR}$')

plt.axhline(y=0.0, color='black', linestyle='-')

plt.errorbar(lTT, 100.*(DlTT_mean/fTT_lcdm(lTT)-1), yerr=100.*DlTT_error_plus/fTT_lcdm(lTT),ls='none', color= 'gray',fmt='o',markersize=3.)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)

plt.show()
plt.clf()


#%% PLOT CMB EE residuals

fig, axes = plt.subplots()

plt.ylim([-100.*0.1,100.*0.1])
plt.xlim([2,2600])

plt.xlabel(r'$\ell$',fontsize=20)
plt.ylabel(r'relative difference $C_{\ell}^{\mathrm{EE}}$ in \%',fontsize=20)

plt.plot(ll_lcdm,100.*(fEE_wzdr(ll_lcdm)/fEE_lcdm(ll_lcdm)-1),'red',linestyle='dashed',label=r'$\mathrm{WZDR}$')
plt.plot(ll_lcdm,100.*(fEE_idm_wzdr(ll_lcdm)/fEE_lcdm(ll_lcdm)-1),'green',label=r'$\mathrm{IDM}-\mathrm{WZDR}$')

plt.axhline(y=0.0, color='black', linestyle='-')
plt.errorbar(lEE, 100.*(DlEE_mean/fEE_lcdm(lEE)-1.), yerr=100.*DlEE_error_plus/fEE_lcdm(lEE),ls='none', color = 'gray',fmt='o',markersize=3.)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)


plt.show()
plt.clf()

#%%  PLOT CMB phi-phi residuals

fig, axes = plt.subplots()

plt.ylim([-100.*0.1,100.*0.1])
plt.xlim([2,2600])

plt.xlabel(r'$\ell$',fontsize=20)
plt.ylabel(r'relative difference $C_{\ell}^{\phi\phi}$ in \%',fontsize=20)

plt.semilogx(ll_lcdm,100.*(fpp_wzdr(ll_lcdm)/fpp_lcdm(ll_lcdm)-1),'red',linestyle='dashed',label=r'$\mathrm{WZDR}$')
plt.semilogx(ll_lcdm,100.*(fpp_idm_wzdr(ll_lcdm)/fpp_lcdm(ll_lcdm)-1),'green',label=r'$\mathrm{IDM}-\mathrm{WZDR}$')

plt.axhline(y=0.0, color='black', linestyle='-')

l_pp_1 =np.linspace(8,20,10)
l_pp_2 =np.linspace(20,39,10)
l_pp_3 =np.linspace(39,65,10)
l_pp_4 =np.linspace(65,100,10)
l_pp_5 =np.linspace(101,144,10)
l_pp_6 =np.linspace(145,198,10)
l_pp_7 =np.linspace(199,263,10)
l_pp_8 =np.linspace(264,338,10)
l_pp_9 =np.linspace(339,425,10)
l_pp_10=np.linspace(426,525,10)
l_pp_11=np.linspace(526,637,10)
l_pp_12=np.linspace(638,762,10)
l_pp_13=np.linspace(763,901,10)
l_pp_14=np.linspace(902,2048,10)

plt.fill_between(l_pp_1, -100.*0.2/1.24, 100.*0.2/1.24, color='lightgray' )
plt.fill_between(l_pp_2, -100.*0.11/1.40,100.*0.11/1.40, color='lightgray' )
plt.fill_between(l_pp_3, -100.*0.08/1.34,100.*0.08/1.34, color='lightgray' )
plt.fill_between(l_pp_4, -100.*0.05/1.14,100.*0.05/1.14, color='lightgray' )
plt.fill_between(l_pp_5, -100.*0.05/0.904,100.*0.05/0.904, color='lightgray' )
plt.fill_between(l_pp_6, -100.*0.06/0.686,100.*0.06/0.686, color='lightgray' )
plt.fill_between(l_pp_7, -100.*0.08/0.513,100.*0.08/0.513, color='lightgray' )
plt.fill_between(l_pp_8, -100.*0.10/0.382,100.*0.10/0.382, color='lightgray' )
plt.fill_between(l_pp_9, -100.*0.13/0.285,100.*0.13/0.285, color='lightgray' )
plt.fill_between(l_pp_10,-100.*0.14/0.213,100.*0.14/0.213, color='lightgray' )
plt.fill_between(l_pp_11,-100.*0.19/0.160,100.*0.19/0.160, color='lightgray' )
plt.fill_between(l_pp_12,-100.*0.23/0.121,100.*0.23/0.121, color='lightgray' )
plt.fill_between(l_pp_13,-100.*0.28/0.0934,100.*0.28/0.0934, color='lightgray' )
plt.fill_between(l_pp_14,-100.*0.30/0.0518,100.*0.30/0.0518, color='lightgray' )

plt.tick_params(axis="x", labelsize=18)
plt.tick_params(axis="y", labelsize=18)

plt.show()
plt.clf()