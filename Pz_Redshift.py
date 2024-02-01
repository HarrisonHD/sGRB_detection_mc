import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

tmin = 20e-3
# tmin = 100e-3
# tmin = 1000e-3
# tmin = 3000e-3
# tmin = 5000e-3

# Rflat, Rm1, Rm2, Rm3, Rh1, Rh2, Rh3, Rf, Rf2, Rw, Rs, Rd, Rb, Rn, Rc1, Rc2
sfr_model = 'Rm1'
tmin_value = tmin

filename = f"Pz/Pz_{sfr_model}_{int(tmin_value * 1e3)}.txt"


def Rcf(z):
    sfr_models = {
        'Rflat': Rflat,
        'Rm1': Rm1,
        'Rm2': Rm2,
        'Rm3': Rm3,
        'Rh1': Rh1,
        'Rh2': Rh2,
        'Rh3': Rh3,
        'Rf': Rf,
        'Rf2': Rf2,
        'Rw': Rw,
        'Rs': Rs,
        'Rd': Rd,
        'Rb': Rb,
        'Rn': Rn,
        'Rc1': Rc1,
        'Rc2': Rc2
    }

    try:
        return sfr_models[sfr_model](z)
    except KeyError:
        print("Invalid SFR model. Please choose a valid model.")
        return None


# General Constants
c = 2.99792458e10
G = 6.67259e-8
Mpc = 3.085e24
year = 3.156e7
Myr = 1e6 * year
Gyr = 1e9 * year
Msolar = 1.989e33

# Cosmology

Omega_m = 0.272
Omega_v = 0.728
h0 = 0.704
H0Mpc = h0 * 1e7
H0 = h0 * 3.24e-18
rho = 3 * H0 ** 2 / (8 * np.pi * G)


def Ez(z):
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_v)


def rbar(z):
    return quad(lambda z1: 1 / Ez(z1), 0, z)[0]


def r(z):
    return quad(lambda z1: (c / H0) / Ez(z1), 0, z)[0]


def dLMpc(z):
    return quad(lambda z1: (1 + z) * (c / H0Mpc) / Ez(z1), 0, z)[0]


def dVbar(z):
    return rbar(z) ** 2 / Ez(z)


def dV(z):
    return 4 * np.pi * (c / H0) ** 3 * rbar(z) ** 2 / Ez(z)


def dVMpc(z):
    return 4 * np.pi * (c / H0Mpc) ** 3 * rbar(z) ** 2 / Ez(z)


def dtobar(z):
    return 1 / ((1 + z) * Ez(z))


def dto(z):
    return 1 / (H0 * (1 + z) * Ez(z))


def tobar(z1, z2):
    return quad(lambda z: dtobar(z), z1, z2)[0]


def to(z1, z2):
    return quad(lambda z: dtobar(z) / H0, z1, z2)[0]


# Star formation rates
def Rflat(z):
    return 1


# Madau & Pozzetti 2000


def Rm1(z):
    return 0.3 * (h0 / 0.65) * Ez(z) / (1 + z) ** 1.5 * np.exp(3.4 * z) / (45 + np.exp(3.8 * z))


# Steidel et al. 1999


def Rm2(z):
    return 0.15 * (h0 / 0.65) * Ez(z) / (1 + z) ** 1.5 * np.exp(3.4 * z) / (22 + np.exp(3.4 * z))


# Blain et al. 1999


def Rm3(z):
    return 0.2 * (h0 / 0.65) * Ez(z) / (1 + z) ** 1.5 * np.exp(3.05 * z - 0.4) / (15 + np.exp(2.93 * z))


# Hopkins & Beacom 2006
def Rh1(z):
    return h0 * (0.017 + 0.13 * z) / (1 + (z / 3.3) ** 5.3)


# Baldry et al. IMF
def Rh2(z):
    return h0 * (0.0118 + 0.08 * z) / (1 + (z / 3.3) ** 5.2)


# Li 2008
def Rh3(z):
    return h0 * (0.0157 + 0.118 * z) / (1 + (z / 3.23) ** 4.66)


# Fardal 2008
def Rf(z):
    return (0.0103 + 0.088 * z) / (1 + (z / 2.4) ** 2.8)


def Rf2(z):
    return h0 * 9e8 * 3.24e-18 * year * 0.075 * 3.7 * 0.84 * Ez(z) * (1 + z) ** 3.7 / (
            1 + 0.075 * (1 + z) ** 3.7) ** 1.84


# Wilken 2008
def Rw(z):
    return (0.014 + 0.11 * z) / (1 + (z / 1.4) ** 2.2)


# Springel 2002
alpha = 0.6
beta = 14.0 / 15
zm = 5.4
rom = 0.15


def Rs(z):
    return rom * beta * np.exp(alpha * (z - zm)) / (beta - alpha + alpha * np.exp(beta * (z - zm)))


# Naganime 2006
def tgyr(z):
    return 13.5 - quad(lambda z1: dtobar(z1) / H0 / Gyr, 0, z)[0]


def Rd(z):
    return 0.056 * (tgyr(z) / 4.5) * np.exp(-tgyr(z) / 4.5)


def Rb(z):
    return 0.198 * (tgyr(z) / 1.5) * np.exp(-tgyr(z) / 1.5)


def Rn(z):
    return Rd(z) + Rb(z)


# Rome
dataRome = np.loadtxt("SFR_Rome.txt")
RR = interp1d(dataRome[:, 0], dataRome[:, 1], kind='linear')


# New Bressan Model
def Rc1(z):
    return 0.146 * 2.8 * np.exp(2.46 * (z - 1.72)) / (2.8 - 2.46 + 2.46 * np.exp(2.8 * (z - 1.72)))


def Rc2(z):
    return 0.178 * 2.37 * np.exp(1.8 * (z - 2)) / (2.37 - 1.8 + 1.8 * np.exp(2.37 * (z - 2)))


# SFR vs Cosmic Time in the Observer Frame
def tcgyr(z):
    return quad(lambda z1: dtobar(z1) / H0 / Gyr, 0, z)[0]


tfmax = tcgyr(15)


def Rzf(z):
    return Rcf(z) / (1 + z)


# Generating the data
dataf = np.array([[tcgyr(z), Rzf(z)] for z in np.arange(0, 20, 0.1)])
Rtf = interp1d(dataf[:, 0], dataf[:, 1], kind='linear', bounds_error=False, fill_value=0)

# Probability Distribution of the Coalescence Time
tmax = 13.5
coef = 1

B, _ = quad(lambda t: 1 / (t ** coef), tmin, tmax)
B = 1 / B


def Pt(t):
    return (B / t ** coef) if tmax > t > tmin else 0


# cosmic coalescence rate vs cosmic time in the observer frame
def tsup(t):
    return tmax if (t + tmax) <= tfmax else tfmax - t


def Rtc(t):
    def integrand(t1):
        t_plus_t1 = t + t1
        if t_plus_t1 > dataf[-1, 0]:
            return 0
        else:
            return Rtf(t_plus_t1) * Pt(t1)

    return quad(integrand, tmin, tsup(t), limit=100, epsabs=1e-3, epsrel=1e-3)[0]


# cosmic coalescence rate vs redshift in the observer frame;
def Rzc(z):
    return Rtc(to(0, z) / Gyr)


datac = np.array([[z, Rzc(z)] for z in np.arange(0, 10.1, 0.1)])
Rcc = interp1d(datac[:, 0], datac[:, 1], kind='linear')


# final calculations and export
def Rz(z):
    return Rcc(z) / Rcc(0) / Myr * dVMpc(z)


K, _ = quad(lambda z: Rz(z), 0, 2, limit=100, epsabs=1e-3, epsrel=1e-3)
K = 1 / K


def Pz(z):
    return K * Rz(z)


data = np.array([[round(z, 5), Pz(z)] for z in np.arange(0, 10.01, 0.01)])
np.savetxt(filename, data, fmt='%0.015f')
