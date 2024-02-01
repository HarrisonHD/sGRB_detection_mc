import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd

np.seterr(all="ignore")
data = np.loadtxt('Pz/Rh1/Pz_Rh1_20.txt')


def prior_redshift(z, data):
    z_values = data[:, 0]
    p_values = data[:, 1]

    # Interpolate the probability for the given z
    p_z = np.interp(z, z_values, p_values)

    return p_z


def log_prior_redshift(z, data):
    # Cubic spline interpolation
    spline = interp1d(data[:, 0], data[:, 1], kind='cubic')

    return np.log(spline(z))


def log_prior_sky_position(ra, dec):
    return np.log(np.cos(dec) / (2 * np.pi)) if 0 <= ra <= 2 * np.pi and -np.pi / 2 <= dec <= np.pi / 2 else -np.inf


def log_prior_polarization_angle(psi):
    return np.log(1 / (2 * np.pi)) if 0 <= psi <= 2 * np.pi else -np.inf


def log_prior_cosine_inclination_angle(cos_i):
    return np.log(0.5) if -1 <= cos_i <= 1 else -np.inf


def log_prior_time_interval(tau, lambda_val):
    return np.log((1 / lambda_val) * np.exp(-tau / lambda_val)) if tau >= 0 else -np.inf


def log_prior_mass(m, m_ns=1.4, m_bh=10):
    return 0 if m == m_ns or m == m_bh else -np.inf


def log_prior_beaming_angle(theta):
    return np.log(1 / 25) if 5 <= theta <= 30 else -np.inf


def log_prior_peak_luminosity(L_p):
    epsilon = 1e-90  # small constant to avoid division by zero
    if Lstar / abs(delta1) < L_p <= Lstar:
        return np.log(((L_p / Lstar) ** alpha) + epsilon)
    elif Lstar < L_p <= delta2 * Lstar:
        return np.log(((L_p / Lstar) ** beta) + epsilon)
    else:
        return -np.inf


def log_prior_log_duration(log_T_i, mu=-0.458, sigma=0.502):
    return np.log((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((log_T_i - mu) / sigma) ** 2))


def log_prior(params):
    z, Lp, logTi = params

    lp_z = log_prior_redshift(z, data) if zMin <= z <= zMax else -np.inf
    lp_Lp = log_prior_peak_luminosity(Lp) if 1. / delta1 <= Lp <= delta2 * Lstar else -np.inf
    lp_logTi = log_prior_log_duration(logTi, logTMean,
                                      logTSigma) if logTMean - 3 * logTSigma <= logTi <= logTMean + 3 * logTSigma else -np.inf

    result = lp_z + lp_Lp + lp_logTi


    return result


def log_likelihood(params, data):
    mu = np.mean(data)
    sigma = np.std(data)

    return -0.5 * np.sum((data - mu) ** 2 / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_probability(params, data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)


zMin = 0  # Min redshift
zMax = 10  # Max redshift
PzMax = 1  # Max probability for  the redshift
delta1 = 100
delta2 = 10
alpha = -0.6
beta = -2
Lstar = 5e50
logTMean = -0.458
logTSigma = 0.502

redshift_mean = 1.70
redshift_std = 0.61
luminosity_mean = 10e49
luminosity_std = 10e50

ndim = 3  # Number of parameters
nwalkers = 100  # Number of walkers
nsteps = 10000  # Number of steps
burn_in = 8000  # Burn in period to discard
thin = 10  # nth number of points to keep

mean_ra = np.pi  # Mean Right Ascension
mean_dec = 0  # Mean Declination
mean_psi = np.pi  # Mean Polarization Angle
mean_cos_i = 0  # Mean Cosine Inclination Angle
mean_mass = 1.4
mean_theta = 17.5 * np.pi / 180  # Mean Beaming Angle in radians
mean_logTi = -0.458  # Mean Log Duration

mean_values = [redshift_mean, luminosity_mean, mean_logTi, mean_ra, mean_dec, mean_psi, mean_cos_i, mean_mass]
std_devs = [1 * redshift_std, 1 * luminosity_std, 1 * logTSigma, np.pi, np.pi / 2, np.pi, 1, 5]

# Define bounds for each parameter
redshift_bounds = (0, 10)
luminosity_bounds = (10 ** 40, np.inf)
ra_bounds = (0, 2 * np.pi)
dec_bounds = (-np.pi / 2, np.pi / 2)
psi_bounds = (0, 2 * np.pi)
cos_i_bounds = (-1, 1)
tau_bounds = (0, np.inf)  # Assuming tau should be non-negative
mass_bounds = (1.4, 10)  # Assuming mass is between 1.4 and 10
theta_bounds = (5 * np.pi / 180, 30 * np.pi / 180)  # Beaming angle in radians
log_duration_bounds = (-np.inf, np.inf)

bounds = [redshift_bounds, luminosity_bounds, log_duration_bounds, ra_bounds, dec_bounds, psi_bounds, cos_i_bounds,
          mass_bounds]


# The Function to generate initial points within each of teh bounds
def generate_initial_point(nwalkers, ndim):
    p0 = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        for j in range(ndim):
            valid = False
            while not valid:
                value = np.random.normal(mean_values[j], std_devs[j])
                if bounds[j][0] <= value <= bounds[j][1]:
                    p0[i, j] = value
                    valid = True
    return p0


# Generate initial points
p0 = generate_initial_point(nwalkers, ndim)

# p0 = np.random.normal([redshift_mean, luminosity_mean, logTMean], [redshift_std, luminosity_std, logTSigma], (nwalkers, ndim))


# Plot starting positions of the walkers on the Redshift x Luminosity Plane
plt.figure(figsize=(10, 6))
plt.scatter(p0[:, 0], p0[:, 1])
plt.xlabel('R')
plt.ylabel('L')
plt.title('Luminosity vs Redshift: Starting Points')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xlim([10 ** -1, 10 ** 1])
plt.show()

# sampler without log_prob
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prior)

# Sampler with log_prob
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data,))

# Run the MCMC
for i in tqdm(range(nsteps)):
    sampler.run_mcmc(p0, 1)

samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)

# Get redshift and luminosity values
redshifts = samples[:, 0]
luminosities = samples[:, 1]

# Create the graph
plt.figure(figsize=(10, 6))
plt.scatter(redshifts, luminosities, s=1)
plt.xlabel('Redshift (z)')
plt.ylabel('Luminosity (Lp)')
plt.title('Luminosity vs Redshift')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xlim([10 ** -1, 10 ** 1])

plt.show()

results = []
N = 2
for k in tqdm(range(N)):  # Add progress bar here
    # Get the kth sample
    sample = samples[k]

    # Redshift sampling
    z = sample[0]  # Assuming 'z' is the first column in your samples
    reject_z = np.random.uniform(0, PzMax)
    p_z = prior_redshift(z, data)
    if reject_z < p_z:
        # If the reject value is less than the prior, keep the sample
        # Otherwise, continue to the next iteration
        continue

    # Peak Luminosity sampling
    Lp = sample[1]  # Assuming 'Lp' is the second column in your samples
    reject_Lp = np.random.uniform(0, 1)  # Assuming a maximum probability of 1 for normalization
    if Lp < 1:
        p_Lp = Lp ** alpha
    else:
        p_Lp = Lp ** beta
    if reject_Lp < p_Lp:
        # If the reject value is less than the prior, keep the sample
        # Otherwise, continue to the next iteration
        continue
    Lp = Lp * Lstar
    logL = np.log10(Lp)

    # Duration sampling
    logTi = sample[2]  # Assuming 'logTi' is the third column in your samples
    reject_logTi = np.random.uniform(0, 1)  # Assuming a normalized Gaussian
    p_logTi = np.exp(-0.5 * ((logTi - logTMean) / logTSigma) ** 2)
    if reject_logTi < p_logTi:
        # If the reject value is less than the prior, keep the sample
        # Otherwise, continue to the next iteration
        continue
    Ti = 10 ** logTi

    # Adjust luminosity based on duration and redshift
    if Ti * (1 + z) < 1:
        logL = logL + logTi + np.log10(1 + z)

    results.append((z, Lp))

# Assuming 'results' is a list of tuples where each tuple is (z, Lp, logTi)
results = [(z, Lp, logTi) for z, Lp, logTi in samples]

# Separate the results into two lists for easier plotting
z_values = [result[0] for result in results]
Lp_values = [result[1] for result in results]

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(z_values, Lp_values, s=1)
plt.xlabel('Redshift (z)')
plt.ylabel('Luminosity (Lp)')
plt.title('Luminosity vs Redshift after reject')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xlim([10 ** -1, 10 ** 1])
plt.show()

# Convert the results to a numpy array for easier exporting
results_np = np.array(results)

# Convert the results to a DataFrame
df = pd.DataFrame(samples, columns=['Redshift', 'Luminosity', 'logTi'])

# Save the DataFrame to a text file with headers
df.to_csv('Data/mcmc_out.txt', index=False)
