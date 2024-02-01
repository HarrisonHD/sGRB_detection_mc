import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd

data = np.loadtxt('Pz_Rh1_20.txt')
z_values, p_values = data[:, 0], data[:, 1]
spline = InterpolatedUnivariateSpline(z_values, p_values, k=4)


def prior(z, min_val, max_val, func):
    return 0 if z < min_val or z > max_val else func(z)


def rejection_sampling_sgrb(N, zMin, zMax, PzMax, delta1, delta2, alpha, beta, Lstar, logTMean, logTSigma, lambda_val,
                            m_ns, m_bh, burn_in, thin):
    results = []
    for k in range(N):
        z = prior(np.random.uniform(zMin, zMax), zMin, zMax, spline)
        ra = prior(np.random.uniform(0, 2 * np.pi), 0, 2 * np.pi, lambda ra: np.cos(ra) / (2 * np.pi))
        dec = prior(np.random.uniform(-np.pi / 2, np.pi / 2), -np.pi / 2, np.pi / 2,
                    lambda dec: np.cos(dec) / (2 * np.pi))
        psi = prior(np.random.uniform(0, 2 * np.pi), 0, 2 * np.pi, lambda psi: 1 / (2 * np.pi))
        cos_i = prior(np.random.uniform(-1, 1), -1, 1, lambda cos_i: 0.5)
        tau = prior(np.random.exponential(scale=lambda_val), 0, np.inf,
                    lambda tau: (1 / lambda_val) * np.exp(-tau / lambda_val))
        m = prior(np.random.choice([m_ns, m_bh]), m_ns, m_bh, lambda m: 1 if m == m_ns or m == m_bh else 0)
        theta = prior(np.random.uniform(5, 30), 5, 30, lambda theta: 1 / 25)
        Lp = prior(np.random.uniform(1. / delta1, delta2), 1. / delta1, delta2,
                   lambda Lp: Lp ** alpha if Lp < 1 else Lp ** beta)
        logTi = prior(np.random.uniform(logTMean - 3 * logTSigma, logTMean + 3 * logTSigma), logTMean - 3 * logTSigma,
                      logTMean + 3 * logTSigma, lambda logTi: np.exp(-0.5 * ((logTi - logTMean) / logTSigma) ** 2))
        if k >= burn_in and k % thin == 0:
            results.append((z, Lp, ra, dec, psi, cos_i, tau, m, theta))
    return results


samples = rejection_sampling_sgrb(10000, 0, 10, 1, 100, 10, -0.6, -2, 10e51, -0.458, 0.502, 1, 1.4, 10, 9000, 1)

plt.figure(figsize=(15, 10))
redshifts, luminosities = zip(*[(sample[0], sample[1]) for sample in samples])
plt.scatter(redshifts, luminosities, s=1)
plt.xlabel('Redshift (z)')
plt.ylabel('Luminosity (Lp)')
plt.title('Luminosity vs Redshift')
plt.grid(True)
plt.yscale('log')
plt.show()

df = pd.DataFrame(samples,
                  columns=['Redshift', 'Luminosity', 'RA', 'Dec', 'Polarization Angle', 'Cosine of Inclination Angle',
                           'Time Interval', 'Mass', 'Beaming Angle'])
df.to_csv('Data/Rejection_out.txt', index=False)

#%%
