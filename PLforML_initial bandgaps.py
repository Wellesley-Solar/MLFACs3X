#should be expanded into a juptyer notbook that goes over:
#importing data
#correcing for integration
#pulling time
#plot
#fit
#plot dynamic behavior
#%%
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
sys.path.append('/Users/rbelisle/Documents/GitHub/PL_Tools')
from PLfunctions import back_subtract, sample_name, find_nearest, weighted_PL, trim_data

def gaussian(x, a, b, c): 
    #generic gaussian curve, for XRD analysis:
    #x is a 1D array of two theta or q values
    #a is the max intensity of the peak and representative of crystalling
    #b is the peak position and 
    # c is the FWHM
    return a*np.exp(-(x - b)**2/(2*c**2))

def wavetoev(wave):
    c_light = 2.99792458e8 * 1e9 #
    h_plank = 4.1357e-15 
    return h_plank*c_light/wave

#%% set up details of experiment and process data into a single dataframe
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/Fall 2021 New Triple Halide Perovskite Search/211223_round2_PL' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

# %% Fit first frame with gaussian peak to detemine initial bandgap
chemistry = []
center = []
uncertainty = []
lim1 = 600 #high energy cutoff
lim2 = 750 #low energy cutoff
#fig1,ax1 = plt.subplots()
i = 0
limits = [[600,700],[600,700],[600,720],[600,720],[640,750],[600,750],[600,750],[600,750],[600,680]]
guess = [[80,665,15],[10,660,1],[100,680,2],[1500,705,15],[2000,720,10],[500,680,10],[1000,690,10],[1000,690,10],[10,630,5]]

for each in all_files:
    df = pd.read_csv(each)
    chem = each.split('_')[-1].split('.')[0]

    first = df[df['Frame'] == 1]['Intensity'] #read file
    first_wave = df[df['Frame'] == 1]['Wavelength'] 
    wave_cut, PL_cut = trim_data(first_wave,first,limits[i][0],limits[i][1])
    PL_back = PL_cut - np.mean(PL_cut[0:10])
    fig1,ax1 = plt.subplots()
    fit, error = curve_fit(gaussian,wave_cut,PL_back, p0=guess[i])
    ax1.plot(wave_cut,PL_back,'b-', label = chem)
    ax1.plot(wave_cut,gaussian(wave_cut,*fit),'k--')
    chemistry.append(chem)
    center.append(fit[1])
    uncertainty.append(np.sqrt(np.diag(error))[1])
    #ax1.legend(loc='lower left')
    i = i+1

pl_bandgap = []
pl_uncertainty = []
i = 0
for each in center:
    pl_bandgap.append(wavetoev(each))
    pl_uncertainty.append(wavetoev(each-uncertainty[i])-wavetoev(each))
    i = i+1

print(pl_bandgap,pl_uncertainty)
pl_bandgap = np.concatenate((np.array(pl_bandgap[0]),np.array(pl_bandgap[2:])), axis=None)
pl_uncertainty = np.concatenate((np.array(pl_uncertainty[0]),np.array(pl_uncertainty[2:])), axis=None)


#%% Comparisson of initial bandgaps and uncertainties 
tauc_bandgap = [1.8, 1.79, 1.73, 1.7, 1.84, 1.74, 1.77, 2]
tauc_uncertainty = [.06, .07, .04, .08, .09, .07, .06, .1]
chems = [11, 12, 13, 14, 15, 16, 17, 18]

fig2,ax2 = plt.subplots(figsize=(6, 4), dpi=400)
ax2.errorbar(chems,tauc_bandgap,tauc_uncertainty, fmt='rs', capsize=6, barsabove=True, label = 'Tauc Fit' )
ax2.errorbar(chems,pl_bandgap,pl_uncertainty, fmt='bs', capsize=6, barsabove=True, label = 'PL Fit' )
ax2.set_xlabel('Chemistry #')
ax2.set_ylabel('Initial Bandgap [eV]')
ax2.legend(loc='upper left')
print(pl_bandgap)
print(pl_uncertainty)

