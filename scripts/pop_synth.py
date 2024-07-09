"""
    pop_synth.py
    Author: William G Lamb
    
    Population synthesis of SMBHBs and calculation of the idealized GWB

    This script contains functions to generate a population synthesis of
    SMBHBs and calculate the idealized GWB characteristic strain squared
    (Phinney 2001). The script also contains functions to calculate the
    mean, variance, skewness and kurtosis of the GWB from a population
    synthesis (Lamb & Taylor 2024).

    The script can be run as a standalone script to generate a population
    synthesis of the GWB and save the results to a file.
"""

import argparse
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# CONSTANTS
G = 4.517103 * 10**-48  # Mpc3 / Msun s2
c = 9.7156119 * 10**-15  # Mpc / s


def smbhb_density(log10_M: float, z: float, alpha: float, log10_M_star: float,
                  beta: float, z0: float, n0_dot: float) -> float:
    """
        SMBHB number density per unit comoving volume as a function of chirp
        mass and redshift

        Parameters
        ----------
        z : float
            Redshift
        log10_M : float
            log10 of the chirp mass
        alpha : float
            Slope of the mass function
        log10_M_star : float
            Characteristic mass of the mass function
        beta : float
            Slope of the redshift function
        z0 : float
            Characteristic redshift of the redshift function
        n0_dot : float
            Normalization of the number density

        Returns
        -------
        float
            log10 SMBHB number density per unit log10_M and unit redshift
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # standard cosmology

    # transform number density normalisation to the right units
    n0_dot *= u.Mpc**-3 * u.Gyr**-1
    n0_dot = n0_dot.to(1/(u.s * u.Mpc**3)).value

    # mass function - compute in base 10 then raise to the power of 10 to
    # avoid numerical issues
    log10M_dist = 10**-(alpha*(log10_M - 7.) +
                        10**(log10_M - log10_M_star) * np.log10(np.e))

    z_dist = (1+z)**beta * np.exp(-z/z0)  # redshift function

    # change in age of binary per unit redshift
    dt_dz = 1/((1+z) * cosmo.H(z).to('Mpc / Mpc s').value)

    return n0_dot * log10M_dist * z_dist * dt_dz


def gwb_ideal(f: np.ndarray, M: np.ndarray, z: np.ndarray,
              model_params: dict) -> float:
    """
        Idealized GW characteristic strain squared (Phinney 2001)

        Parameters
        ----------
        f : np.ndarray
            Frequency array
        log10_M : np.ndarray
            log10 of the chirp mass array
        z : np.ndarray
            Redshift array
        model_params : dict
            Model parameters

        Returns
        -------
        hc2 : float
            idealized GW characteristic strain squared
    """
    log10_M_3d = np.log10(M)[None, :, None]
    z_3d = z[None, None, :]

    dn = smbhb_density(log10_M_3d, z_3d, **model_params)

    integrand = (1+z_3d)**(-1/3) * (10**log10_M_3d)**(5/3) * dn

    # integrate over log10_M and z with trapezoidal rule
    dlog10_M = log10_M_3d[0, 1, 0] - log10_M_3d[0, 0, 0]
    dz = z[1] - z[0]
    integral = ((integrand[:, 1:, 1:] + integrand[:, :-1, :-1])
                * dlog10_M * dz / 4).sum(axis=(1, 2))

    prefactor = 4*G**(5/3) / (3 * np.pi**(1/3) * c**2 * f**(4/3))

    return prefactor * integral


def h2_f(f: float|np.ndarray, M: float|np.ndarray, z: float|np.ndarray,
         ) -> np.ndarray|float:
    """
        Squared GW strain amplitude for a circular SMBHB

        Parameters
        ----------
        f : float or np.ndarray
            Frequency
        M : float or np.ndarray
            Chirp mass
        z : float or np.ndarray
            Redshift
        
        Returns

    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl2 = cosmo.comoving_distance(z).to(u.Mpc).value ** -2.
    fr = f * (1+z)
    h2 = 32/5 * c**-8. * (G * M)**(10/3) * dl2 * (np.pi*fr)**(4/3)

    return h2

def smbhb_number_per_cell(freqs: np.ndarray, M_edges: np.ndarray,
                          z_edges: np.ndarray, model_params: dict,
                          freq_lambda: float = -11/3, return_int: bool = True,
                          ) -> np.ndarray:
    """
    The number of SMBHBs per grid cell of frequency, chirp mass and redshift

    Assumes standard cosmology with H0=70, Om0=0.3

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array
    M_edges : np.ndarray
        Chirp mass edges
    z_edges : np.ndarray
        Redshift edges
    model_params : dict
        Model parameters
    freq_lambda : float
        Frequency power law
    return_int : bool
        Return the number of SMBHBs per grid cell if True. Otherwise, return
        the differential number per unit frequency, chirp mass and redshift.
    
    Returns
    -------
    np.ndarray
        Number of SMBHBs per (f, log10M, z) grid cell

    """
    # reshaping arrays
    f_3d = 0.5 * (freqs[1:] + freqs[:-1])[:, None, None]
    M_3d = M_edges[None, :, None]
    log10_M_3d = np.log10(M_3d)
    z_3d = z_edges[None, None, :]

    # differentials
    dlogf = (np.log(freqs[1:]) - np.log(freqs[:-1]))[:, None, None]
    dlog10_M = (np.log10(M_edges[1:]) - np.log10(M_edges[:-1]))[None, :, None]
    dz = z_edges[1] - z_edges[0]

    # number density of SMBHBs per unit comoving volume
    d2n = smbhb_density(log10_M_3d, z_3d, **model_params)

    fr = f_3d * (1+z_3d)  # frequency in the rest frame of the binary

    # differential of the time to coalescence with respect to log10(f_r)
    dt_dlogfr = ((5/96) * (G * M_3d / c**3)**(-5/3) *
                 (np.pi * fr)**(freq_lambda+1))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl2 = cosmo.comoving_distance(z_3d).to('Mpc').value ** 2.
    cosmo_factor = 4 * np.pi * c * (1 + z_3d) * dl2

    N = d2n * dt_dlogfr * cosmo_factor

    if return_int:
        trapz = (N[:, 1:, 1:] + N[:, :-1, :-1]) * dlog10_M * dz / 4
        trapz *= dlogf

        return trapz

    else:
        return N


def pop_synth(N: np.ndarray, freqs: np.ndarray, M_mid: np.ndarray,
              z_mid: np.ndarray, n_real=100000, seed=None) -> np.ndarray:
    """
        Population synthesis of SMBHBs

        Parameters
        ----------
        N : np.ndarray
            Number of SMBHBs per grid cell of frequency, chirp mass and
            redshift
        freqs : np.ndarray
            Frequency array
        log10_M_mid : np.ndarray
            Midpoints of the chirp mass bins
        z_mid : np.ndarray
            Midpoints of the redshift bins
        n_real : int
            Number of realisations
        seed : int
            Random seed

        Returns
        -------
        np.ndarray
            Realisations of the GWB characteristic strain squared
    """
    dlogf = np.log(freqs[1:]) - np.log(freqs[:-1])
    size = (freqs.shape[0], M_mid.shape[0], z_mid.shape[0])
    h2cf = np.zeros((n_real, freqs.shape[0]))

    # random number generator
    rng = np.random.default_rng(seed=seed)

    # generate realisations
    for nreal in range(n_real):
        s = rng.poisson(lam=N, size=size)
        h2 = h2_f(freqs[:, None, None], M_mid[None, :, None],
                  z_mid[None, None, :])
        h2cf[nreal, :] = (s * h2).sum(axis=(-2, -1)) / dlogf

    return h2cf

def ideal_gwb_stats(N: np.ndarray, freqs: np.ndarray, log10_M_mid: np.ndarray,
                    z_mid: np.ndarray, dlogf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Compute the idealised mean, variance, skewness and kurtosis of the GWB
        from a population synthesis (Lamb & Taylor 2024)

        Parameters
        ----------
        N : np.ndarray
            Number of SMBHBs per grid cell of frequency, chirp mass and
            redshift
        freqs : np.ndarray
            Frequency array
        log10_M_mid : np.ndarray
            Midpoints of the chirp mass bins
        z_mid : np.ndarray
            Midpoints of the redshift bins
        dlogf : np.ndarray
            Bin widths of the log10 of the frequency

        Returns
        -------
        tuple
            Mean, variance, skewness and kurtosis of the GWB
    """
    h2 = h2_f(freqs[:, None, None], 10**log10_M_mid[None, :, None],
              z_mid[None, None, :])
    
    mean_h2cf = (N * h2).sum(axis=(-2, -1)) / dlogf
    var_h2cf = (N * h2**2).sum(axis=(-2, -1)) / dlogf**2
    skew_h2cf = (N * h2**3).sum(axis=(-2, -1)) / dlogf**3 / var_h2cf**1.5
    kurt_h2cf = (N * h2**4).sum(axis=(-2, -1)) / dlogf**4 / var_h2cf**2

    return mean_h2cf, var_h2cf, skew_h2cf, kurt_h2cf


def bootstrap(n: int, seed: int = None) -> np.ndarray:
    """
        A function to generate bootstrap indices

        Parameters
        ----------
        n : int
            Number of indices to generate
        seed : int
            Random seed

        Returns
        -------
        np.ndarray
            Bootstrap indices
    """
    rng = np.random.default_rng(seed=seed)
    return rng.choice(n, n, replace=True)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--freq_lambda', type=float, default=-11/3)
    argparser.add_argument('--Nidx', type=int, default=0)
    argparser.add_argument('--outdir', type=str, default='./data')
    args = argparser.parse_args()

    # defining model
    model4 = dict(alpha=-0.5, log10_M_star=np.log10(4.2*10**8), beta=1.5,
                  z0=1.1, n0_dot=9e-5)

    # defining grid cells
    TSPAN = 20 * 365.24 * 86400
    gw_fbins = (np.arange(1, 32) - 0.5) / TSPAN
    gw_freqs = 0.5 * (gw_fbins[1:] + gw_fbins[:-1])

    log10_chirp_mass_bins = np.linspace(6, 11, 1001)
    log10_chirp_mass_mid = 0.5 * (log10_chirp_mass_bins[1:] +
                                  log10_chirp_mass_bins[:-1])

    z_bins = np.linspace(0, 5, 101)
    z_mid = 0.5 * (z_bins[1:] + z_bins[:-1])

    # generating populations synthesis
    N = smbhb_number_per_cell(gw_freqs, 10**log10_chirp_mass_mid, z_bins,
                              model4, freq_lambda=args.freq_lambda)

    N_REALISATIONS = 100000
    h2cf = pop_synth(N, gw_freqs, 10**log10_chirp_mass_mid, z_mid,
                     N_REALISATIONS)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(f'{args.outdir}/h2cf_{args.Nidx}.npy', h2cf)
