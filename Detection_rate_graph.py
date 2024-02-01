import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from scipy.integrate import quad
import random


def calculate_luminosity_distance(redshift):
    """
    Calculate the luminosity distance given redshift.
    """
    return cosmo.luminosity_distance(redshift).to(u.cm).value


def add_luminosity_column(data):
    """
    Add a column for luminosity
    """
    data['Luminosity Distance (cm)'] = data['Redshift'].apply(calculate_luminosity_distance)
    data['Fluence (erg/cm^2)'] = data['BAT Fluence (15-150 keV) [10^-7 erg/cm^2]'] * 1e-7
    data['Luminosity (erg/s)'] = 4 * np.pi * data['Luminosity Distance (cm)'] ** 2 * data['Fluence (erg/cm^2)']
    return data


def band(x, alpha, beta, e0):
    diff = alpha - beta
    x = np.array(x) / 100  # normalization at 100 keV
    e0 = e0 / 100
    output = x

    output = (diff * e0) ** diff * np.exp(-diff) * x ** beta
    # Handle both single values and arrays
    if isinstance(x, np.ndarray):
        index = np.where(x < diff * e0)
        output[index] = x[index] ** alpha * np.exp(-x[index] / e0)
    else:
        if x < diff * e0:
            output = x ** alpha * np.exp(-x / e0)
    return output


def D(H, z0, omega):
    c = 299792.458
    H = H / 3.084938E24
    output = (1 + z0) * (eta(1, omega) - eta(1 / (1 + z0), omega))
    if output == 0:
        output = 1
    output *= c / H
    return output


def eta(a, omega):
    s = ((1 - omega) / omega) ** (1 / 3)
    output = 2 * np.sqrt(s ** 3 + 1) * (1 / (a ** 4) - 0.1540 * s / (a ** 3) + 0.4304 * s ** 2 / a ** 2 + 0.19097 * (
            s ** 3) / a + 0.066941 * s ** 4) ** (-1 / 8)
    return output


def is_above_sensitivity_line(redshift, luminosity):
    """
    Check if a point is above the sensitivity line.
    """
    # Find the closest index in the zb array
    index = np.abs(zb - redshift).argmin()
    # Check if the luminosity is greater than the sensitivity line at this redshift
    return luminosity > Luminosity_swift[index]





def sample_points_above_sensitivity_line(data, num_points):
    """
    Randomly sample points from the data until num_points are found that are above the sensitivity line.
    """
    # Initialize a list to store the sampled points
    sampled_points = []
    # Initialize a counter for the total number of points sampled
    total_points_sampled = 0
    # Continue sampling points until enough have been found
    while len(sampled_points) < num_points:
        # Randomly sample a point
        point = data.iloc[random.choice(range(len(data)))]
        # Increment the total number of points sampled
        total_points_sampled += 1
        # Check if the point is above the sensitivity line
        if is_above_sensitivity_line(point['Redshift'], point['Luminosity']):
            # Add the point to the list of sampled points
            sampled_points.append(point)
    # Calculate the detection rate
    detection_rate = num_points / total_points_sampled
    # Return the detection rate and the sampled points as a DataFrame
    return detection_rate, pd.DataFrame(sampled_points)


# Constants
H = 70
omega = 0.3
alpha_min = -2.03
beta_min = -2.3
Eb_min = 46
Flim = 2.5
erg_swift_min = 1.4822e-5
photons_swift_min = 374.08
delta1 = 1000
delta2 = 1000
alpha_mean = -0.63
beta_mean = -1.9
L_star = 0.15917e+51

# Redshift values
zb = np.arange(0.01, 10.01, 0.01)
# Calculate Luminosity
Luminosity_swift = np.zeros(len(zb))
quad_band_min = lambda x: band(x, alpha_min, beta_min, Eb_min)

for j, z in enumerate(zb):
    Luminosity_swift[j] = (Flim * 4 * np.pi * (D(H, z, omega) ** 2) / (1 + z) /
                           quad(quad_band_min, 15, 150)[0] *
                           quad(quad_band_min, 15 / (1 + z), 150 / (1 + z))[0] *
                           erg_swift_min / photons_swift_min)

# Compute sensitivity
sensitivity_swift = np.zeros(len(zb))
for j, L in enumerate(Luminosity_swift):
    if L_star / delta1 < L <= L_star:
        sensitivity_swift[j] = quad(lambda x: (x / L_star) ** alpha_mean, L, L_star)[0]
    elif L_star < L <= delta2 * L_star:
        sensitivity_swift[j] = quad(lambda x: (x / L_star) ** beta_mean, L, L_star * delta2)[0]

# Load the Cleaned Swift Data
filepath = 'Data/combined_grb_table.txt'
grb_data = pd.read_csv(filepath, sep='\t')
grb_data = add_luminosity_column(grb_data)

# Load in the appropriate data depending if emcee ro rejection was used


# Load in the Rejection samples
rejection_data = pd.read_csv('Data/Rejection_out.txt', sep=',', header=0, names=['Redshift', 'Luminosity', 'RA', 'Dec', 'Polarization Angle', 'Cosine of Inclination Angle', 'Time Interval', 'Mass', 'Beaming Angle'])

# Loading in emcee samples
#rejection_data = pd.read_csv('Data/mcmc_out.txt', sep=',', header=0, names=['Redshift', 'Luminosity', 'LogTi'])

# make sure 'Redshift' and 'Luminosity' are floats
rejection_data['Redshift'] = rejection_data['Redshift'].astype(float)
rejection_data['Luminosity'] = rejection_data['Luminosity'].astype(float)

# Plot all of the swift data, the sensitivity curve, and the rejection data
plt.figure(figsize=(10, 6))
plt.scatter(grb_data['Redshift'], grb_data['Luminosity (erg/s)'], color='red', s=5, label='Swift Data Points')
plt.scatter(rejection_data['Redshift'], rejection_data['Luminosity'], color='blue', s=1, label='Rejection Data Points')

# Plot the sensitivity curve
plt.plot(zb, Luminosity_swift, label='Sensitivity Curve', color='black', linewidth=1)

plt.xlabel('Redshift')
plt.ylabel('Luminosity (erg/s)')
plt.title('Redshift vs Luminosity')
# Set the scale of the axis
plt.yscale('log')
plt.xscale('log')
plt.xlim([10 ** -1, 10 ** 1])
plt.grid(True)
plt.legend()
plt.show()

# Randomly sample from the list of sGRBs until N are found that are above the sensitivity line

num_points = 30  # number of detected points

detection_rate, sampled_points = sample_points_above_sensitivity_line(rejection_data, num_points)

# now create plot of the 'Detected' points and the swift points
plt.figure(figsize=(20, 9))

# Swift Points
plt.scatter(grb_data['Redshift'], grb_data['Luminosity (erg/s)'], color='red', s=15, label='Swift Data Points')

# Simulated Points
plt.scatter(rejection_data['Redshift'], rejection_data['Luminosity'], color='black', s=20,
            label='Rejection Data Points')

# "Detected" samples points
plt.scatter(sampled_points['Redshift'], sampled_points['Luminosity'], color='blue', s=25, marker='p',
            label='Detected Points')

# Plot the sensitivity curve
plt.plot(zb, Luminosity_swift, label='Sensitivity Curve', color='black', linewidth=1)

plt.xlabel('Redshift')
plt.ylabel('Luminosity (erg/s)')
plt.title('Redshift vs Luminosity')
plt.yscale('log')
plt.xscale('log')
plt.xlim([10 ** -1, 10 ** 1])
plt.xlim([10 ** 47, 10 ** 53])
plt.grid(True)
plt.legend()
plt.show()
