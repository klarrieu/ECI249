import os
import arcpy
from arcpy.sa import *
import numpy as np
import matplotlib.pyplot as plt

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
path = os.path.dirname(__file__) + '\\'

# inputs:
l_suffixes = ['001k', '005k', '021k', '042k', '084k', '110k']
# recurrence interval for each big discharge
lifespans = [1.0, 1.2, 2.5, 4.7, 12.7, 20.0]
# stable grain size at each big discharge
grain_path = path + 'stable_grains\\'
stable_grain_maps = [Raster(grain_path+'d_cr'+str(suffix)) for suffix in l_suffixes]

# spacing for tau_crit vals to integrate
dx = 0.001
# tau crits to integrate lifespan over
tau_crits = np.arange(0.017, 0.078, dx)
# mean/default tau crit
tau_crit0 = 0.047

# hydraulic HSI (product) at each smol discharge
hsi_path = path + 'HSI\\'
s_suffixes = [530, 700, 880, 1000]
dsi_rass = [Raster(hsi_path+'dsi_ras'+str(suffix)) for suffix in s_suffixes]
vsi_rass = [Raster(hsi_path+'vsi_ras'+str(suffix)) for suffix in s_suffixes]
h_hsis = [dsi_rass[i] * vsi_rass[i] for i in range(len(s_suffixes))]
# probability of each spawning season flow range
ps = [0.1, 0.25, 0.3, 0.35]

# grain sizes to optimize
ds = np.arange(0.05, 0.72, 0.01)
# objective function value array
vals = []


def dot(l1, l2):
    """Dot product between two lists"""
    return sum([e1 * e2 for e1, e2 in zip(l1, l2)])

def tau_crit_pdf(tau_crit, mu=tau_crit0, std=0.01):
    """pdf for tau crit"""
    return 1.0/(np.sqrt(2*np.pi) * std) * np.exp(-(tau_crit - mu)**2 * 1.0/(2 * std**2))

def d_hsi(d):
    """substrate HSI for grain size d"""
    if 0.05 <= d <= 0.1:
        return (d-0.05)*1/0.05
    elif 0.1 < d <= 0.3:
        return 1
    elif 0.3 < d <= 0.35:
        return 1 - (d-0.3)*0.6/0.05
    elif 0.35 < d <= 0.66:
        return 0.4
    elif 0.66 <= d <= 0.71:
        return 0.4 - (d-0.66)*0.4/0.05
    else:
        return 0

# for each D_dist:
for d in ds:
    print('d=%.2f' % d)
    # get expected cHSI:
    c_hsis = [(d_hsi(d)*h_hsi)**(1.0/3) for h_hsi in h_hsis]
    expected_cHSI = dot(ps, c_hsis)
    # get expected lifespan:
    expected_lifespan = 0
    for tau_crit in tau_crits:
        # print('tau_crit=%.4f' % tau_crit) ***
        # adjust stable grain maps: multiply by old tau crit / new tau crit
        adj_stable_grain_maps = [grain_map * tau_crit0/tau_crit for grain_map in stable_grain_maps]
        # compare each pixel to stable grain maps for each discharge ---> create lifespan map (Con statements)
        lifespan_map = sum([Con(stable_grain_maps[i] > d, lifespans[i], 0) for i in range(len(stable_grain_maps))])
        lifespan_map = Con(stable_grain_maps[-1] >= 0, 0, 0) # initialize map of zeros
        for i in range(len(stable_grain_maps)):
            lifespan_map = Con((stable_grain_maps[i] > d) & (lifespan_map == 0), lifespans[i], 0)
        # lifespan_map.save(path+'lifespan_%s.tif' % str(d).replace('.', 'pt')) ***
        # multiply by pdf at the tau crit value
        integrand = lifespan_map * tau_crit_pdf(tau_crit)
        # multiply by dx and add to total integral value
        expected_lifespan += integrand * dx

    # take product of expected cHSI and Lifespan
    expected_product = expected_cHSI * expected_lifespan
    # sum all pixels
    val = arcpy.RasterToNumPyArray(expected_product, nodata_to_value=0).sum()
    # save value to array
    vals.append(val)

# get maximum value
optimal_d, max_val = [(d, val) for d, val in zip(ds, vals) if val == max(vals)][0]
print('optimal grain size: %.2f' % optimal_d)

# plot objective function
fig, ax = plt.subplots(1, 1)
ax.plot(ds, vals, '-o')
ax.set(title='Instream Gravel Injection Assessment', xlabel='Mean grain size (in.)', ylabel='Objective function')
ax.grid()
plt.show()

# plot
