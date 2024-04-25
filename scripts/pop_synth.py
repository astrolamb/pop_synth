import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import argparse
import os

# CONSTANTS
G = 4.517103 * 10**-48  # Mpc3 / Msun s2
c = 9.7156119 * 10**-15  # Mpc / s


def dn_dlog10Mdz(log10_M, z, alpha, log10_M_star, beta, z0, n0_dot):
    """
        SMBHB number density per unit log10_M and unit redshift

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

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # transform number density normalisation to the right units
    n0_dot *= u.Mpc**-3 * u.Gyr**-1
    n0_dot = n0_dot.to(1/(u.s * u.Mpc**3)).value

    log10M_dist = 10**-(alpha*(log10_M - 7.) +
                        10**(log10_M - log10_M_star) * np.log10(np.e))

    z_dist = (1+z)**beta * np.exp(-z/z0)

    # change in age of binary per unit redshift
    dt_dz = 1/((1+z) * cosmo.H(z).to('Mpc / Mpc s').value)

    return n0_dot * log10M_dist * z_dist * dt_dz


def gwb_ideal(f, log10_M, z, model_params):
    """
        Idealized GW characteristic strain squared

        Parameters
        ----------
        log10_M : float
            log10 of the chirp mass
        z : float
            Redshift
        model_params : dict
            Model parameters

        Returns
        -------
        log10_hc2 : float
            log10 of the idealized GW characteristic strain squared
    """
    log10_M_3d = log10_M[None, :, None]
    z_3d = z[None, None, :]

    dn = dn_dlog10Mdz(log10_M_3d, z_3d, **model_params)

    integrand = (1+z_3d)**(-1/3) * (10**log10_M_3d)**(5/3) * dn

    dlog10_M = log10_M[1] - log10_M[0]
    dz = z[1] - z[0]
    integral = ((integrand[:, 1:, 1:] + integrand[:, :-1, :-1])
                * dlog10_M * dz / 4).sum(axis=(1, 2))

    prefact = 4*G**(5/3) / (3 * np.pi**(1/3) * c**2 * f**(4/3))

    return prefact * integral


def hc2_f(f, M, z):
    """
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dL2 = cosmo.comoving_distance(z).to(u.Mpc).value ** -2.
    fr = f * (1+z)
    hc2 = 32/5 * c**-8. * (G * M)**(10/3) * dL2 * (np.pi*fr)**(4/3)

    return hc2


def dN_dlog10Mdzdlogf(f, M, z, model_params, freq_lambda=-11/3,
                      return_int=True, mass_limit=True):
    """
    Assuming M and z are the edges of bins
    """
    f_3d = 0.5*(f[1:] + f[:-1])[:, None, None]
    M_3d = M[None, :, None]
    z_3d = z[None, None, :]

    dlogf = (np.log(f[1:]) - np.log(f[:-1]))[:, None, None]
    dlog10_M = (np.log10(M[1:]) - np.log10(M[:-1]))[None, :, None]
    dz = z[1] - z[0]

    d2n = dn_dlog10Mdz(np.log10(M_3d), z_3d, **model_params)

    fr = f_3d * (1+z_3d)
    dt_dlogfr = (5/96) * (G*M_3d/c**3)**(-5/3) * (np.pi * fr)**(freq_lambda+1)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dA2 = cosmo.comoving_distance(z_3d).to('Mpc').value ** 2.
    cosmofact = 4 * np.pi * c * (1+z_3d) * dA2

    dN_ = d2n * dt_dlogfr * cosmofact

    if return_int:
        dN = (dN_[:, 1:, 1:] + dN_[:, :-1, :-1]) * dlog10_M * dz / 4
        dN *= dlogf
    else:
        dN = dN_

    if mass_limit:
        dN_log10M = dN.sum(axis=2)
        cumtrapz = np.cumsum(dN_log10M[:, ::-1], axis=1)[:, ::-1]

        for ii in range(f.shape[0]-1):
            mask = cumtrapz[ii] < 1
            dN[ii, mask] = 0

    return dN


def pop_synth(N, f_mid, log10_M_mid, z_mid, dlogf, nreals=100000, nfreqs=30):
    """
    """
    rng = np.random.default_rng()
    h2cf = np.zeros((nreals, nfreqs))

    for nreal in range(nreals):
        s = rng.poisson(lam=N, size=(nfreqs, log10_M_mid.shape[0],
                                     z_mid.shape[0]))
        h2 = hc2_f(f_mid[:, None, None],
                   10**log10_M_mid[None, :, None],
                   z_mid[None, None, :])
        h2cf[nreal, :] = (s * h2).sum(axis=(-2, -1)) / dlogf

    return h2cf


def bootstrap(h2cf):
    """
    """
    vars = np.zeros_like(h2cf)
    for ii in range(h2cf.shape[0]):
        idxs = np.random.choice(h2cf.shape[0], h2cf.shape[0], replace=True)
        vars[ii] = np.var(h2cf[idxs], axis=0)
    return vars


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--freq_lambda', type=float, default=-11/3)
    argparser.add_argument('--Nidx', type=int, default=0)
    argparser.add_argument('--outdir', type=str, default='./data')
    args = argparser.parse_args()

    # defining model
    model4 = dict(alpha=-0.5, log10_M_star=np.log10(4.2*10**8), beta=1.5,
                  z0=1.1, n0_dot=9e-5)

    # defining grid
    Tspan = 20 * 365.24 * 86400
    fbins = (np.arange(1, 32) - 0.5) / Tspan
    f_mid = 0.5 * (fbins[1:] + fbins[:-1])

    log10_M_bins = np.linspace(6, 11, 1001)
    log10_M_mid = 0.5 * (log10_M_bins[1:] + log10_M_bins[:-1])

    z_bins = np.linspace(0, 5, 101)
    z_mid = 0.5 * (z_bins[1:] + z_bins[:-1])

    N = dN_dlog10Mdzdlogf(fbins, 10**log10_M_bins, z_bins, model4,
                          mass_limit=False, freq_lambda=args.freq_lambda)

    nreals = 100000
    dlogf = (np.log(fbins[1:]) - np.log(fbins[:-1]))

    h2cf = pop_synth(N, f_mid, log10_M_mid, z_mid, dlogf, nreals, 30)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(f'{args.outdir}/h2cf_{args.Nidx}.npy', h2cf)
