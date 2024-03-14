"""
A script/class to generate synthetic populations of compact binaries
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.special import logsumexp
from scipy.integrate import cumulative_trapezoid

# Constants - in units of Mpc, Msun, s
G = 4.517103 * 10**-48  # Mpc3 / Msun s2
c = 9.7156119 * 10**-15  # Mpc / s

# default PTA timespan
Tspan = 20 * 365.24 * 24 * 3600  # 15 years in seconds

# default cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # flat cosmo, Omega_m = 0.3


class PopSynth:
    """
    Class to generate synthetic populations of compact binaries
    """

    def __init__(self, zmax: float = 5., log10Mmin: float = 6.,
                 log10Mmax: float = 11., zbins: int = 101,
                 log10Mbins: int = 101, Tspan: float = Tspan, Nfreqs: int = 30,
                 cosmo: FlatLambdaCDM = cosmo, model_params: dict = {}):
        """
        Initialize the class with the parameters of the population.
        Currently assumes circular SMBHBs that do not interact with its
        environment.

        Parameters
        ----------
        zmax : float
            Maximum redshift of the population
        log10Mmin : float
            Minimum log10 of the chirp mass of the population
        log10Mmax : float
            Maximum log10 of the chirp mass of the population
        zbins : int
            Number of redshift bins
        log10Mbins : int
            Number of log10 chirp mass bins
        Tspan : float
            Timespan of the PTA observations in seconds
        Nfreqs : int
            Number of frequencies to use in the population
        cosmo : astropy.cosmology
            Cosmology object
        model_params : dict
            Model parameters for the SMBHB number density

        Returns
        -------
        None
        """

        self.zmax = zmax
        self.log10Mmin = log10Mmin
        self.log10Mmax = log10Mmax
        self.zbins = zbins
        self.log10Mbins = log10Mbins
        self.Tspan = Tspan
        self.Nfreqs = Nfreqs
        self.cosmo = cosmo
        self.model_params = model_params

        # initialising frequency bins
        f = np.arange(1, self.Nfreqs+1)/self.Tspan
        fbin_edges = 0.5 * (f[:-1] + f[1:])
        fbin_edges = np.insert(fbin_edges, 0, 0.5*(0.5/Tspan + f[0]))
        fbin_edges = np.append(fbin_edges,
                               0.5*(f[-1] + (self.Nfreqs+1)/self.Tspan))
        self.df = 1/Tspan

        # initialising log10 frequency bins
        self.log10_f = np.log10(f)
        self.log10_fbin_edges = np.log10(fbin_edges)
        self.dlog10_f = self.log10_fbin_edges[1:] - self.log10_fbin_edges[:-1]

        # initialising log10 chirp mass bins
        self.log10_M = np.linspace(self.log10Mmin, self.log10Mmax,
                                   self.log10Mbins)
        self.log10_Mbin_edges = np.linspace(self.log10Mmin, self.log10Mmax,
                                            self.log10Mbins+1)
        self.dlog10_M = self.log10_Mbin_edges[1] - self.log10_Mbin_edges[0]

        # initialising redshift bins
        self.z = np.linspace(0.001, self.zmax, self.zbins)
        self.zbin_edges = np.linspace(0, self.zmax, self.zbins+1)
        self.dz = self.zbin_edges[1] - self.zbin_edges[0]

        # compute mass limit
        self.max_mass = self.mass_limit(self.smbhb_number_per_bin(
            self.log10_f[:, None, None], self.z[None, None, :],
            self.log10_M[None, :, None], model_params
        ))

    def smbhb_number_density(self, z: float, log10_M: float, alpha: float,
                             log10_M_star: float, beta: float, z0: float,
                             n0_dot: float) -> float:
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

        # transform number density normalisation to the right units
        n0_dot *= u.Mpc**-3 * u.Gyr**-1
        n0_dot = n0_dot.to(1/(u.s * u.Mpc**3)).value

        # change in age of binary per unit redshift
        dt_dz = 1/((1+z) * cosmo.H(z).to('Mpc / Mpc s').value)

        # log10 mass function
        mass_dist = (-alpha * (log10_M - 7.) - 10**(log10_M - log10_M_star) *
                     np.log10(np.e))

        # log10 redshift function
        z_dist = (beta * np.log10(1+z) - (z/z0) * np.log10(np.e) +
                  np.log10(dt_dz))

        # log10 number density
        log10_n_dot = np.log10(n0_dot) + mass_dist + z_dist

        return log10_n_dot

    def log10_dE_dfr(self, log10_f: float, log10_M: float, z: float) -> float:
        """
        Change in GW energy emission per unit log10 rest-frame frequency
        Equation (5) of Sato-Polito & Kamionkowski (2023)

        Parameters
        ----------
        log10_f : float
            log10 of the observed frequency
        log10_M : float
            log10 of the chirp mass
        z : float
            Redshift

        Returns
        -------
        float
            log10 of the change in GW energy emission per unit log10 rest-frame
            frequency
        """

        # rest-frame frequency
        log10_fr = log10_f + np.log10(1+z)

        # change in GW energy emission per unit log10 rest-frame frequency
        return (-np.log10(3*G) + (5/3)*(np.log10(G) + log10_M) +
                (2/3)*(np.log10(np.pi) + log10_fr))

    def log10_hc2_ideal(self, log10_M: float, z: float,
                        model_params: dict) -> float:
        """
        Idealized GW characteristic strain squared
        Equation (1) of Sato-Polito & Kamionkowski (2023)

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

        # initialising empty arrays
        log10_hc2 = np.zeros_like(self.log10_f)

        # loop over frequency bins
        for ii in range(self.Nfreqs):

            # effective chirp mass limit
            log10_M_eff = log10_M[log10_M <= self.max_mass[ii]]

            # change in GW energy emission per unit log10 rest-frame frequency
            log10_dE_dfr = self.log10_dE_dfr(self.log10_f[ii],
                                             log10_M_eff[:, None],
                                             z[None, :])

            # smbhb number density per unit log10_M and unit redshift
            log10_n = self.smbhb_number_density(z[None, :],
                                                log10_M_eff[:, None],
                                                **model_params) - np.log10(1+z)

            # integrand
            #integrand = (log10_dE_dfr + log10_n)
            #integrand *= np.log(10)  # log10 to ln

            # log10 of the idealized GW characteristic strain squared
            #log10_hc2[ii] = logsumexp(
            #    integrand[ii], b=self.dlog10_M*self.dz*self.dlog10_f[ii],
            #) / np.log(10)  # ln to log10

            # use trapezoidal rule to integrate over log10_M and z
            integrand = 10**(log10_dE_dfr + log10_n)
            #integrand *= self.dlog10_M*self.dz
            int_over_log10M = np.trapz(integrand, x=log10_M_eff, axis=0)
            log10_hc2[ii] = np.log10(np.trapz(int_over_log10M, x=z))# *
                                     #self.dlog10_f[ii])
            
        log10_hc2 += np.log10(4*G/(np.pi*c**2)) - 2*self.log10_f

        return log10_hc2  #+ self.dlog10_f

    def log10_dlogf_dt(self, log10_f: float, log10_M: float, z: float) -> float:
        """
        Rest-frame GW frequency evolution model
        Equation (4) of Sato-Polito & Kamionkowski (2023)
        """

        log10_fr = np.log10(1+z) + log10_f
        return (np.log10(96/5) + (8/3)*np.log10(np.pi) +
                (5/3)*(np.log10(G) + log10_M) - 5*np.log10(c) +
                (8/3)*log10_fr)

    def smbhb_number_per_bin(self, log10_f: float, z: float, log10_M: float,
                             model_params: dict) -> float:
        """
        SMBHB number per log10 frequency, log10_M and redshift bin

        Parameters
        ----------
        z : float
            Redshift
        log10_M : float
            log10 of the chirp mass
        model_params : dict
            Model parameters

        Returns
        -------
        float
            log10 SMBHB number per log10 frequency, log10_M and redshift bin
        """

        # change in log GW frequency per unit log10 rest-frame frequency
        log10_dlogf_dt = self.log10_dlogf_dt(log10_f, log10_M, z)

        # cosmology factor
        dA = cosmo.angular_diameter_distance(z).to('Mpc').value
        log10_dt_dzdz_dVc = -np.log10(4 * np.pi * c * (1+z) * dA**2)

        # log10 number density
        log10_n = self.smbhb_number_density(z, log10_M, **model_params)

        return log10_n - log10_dlogf_dt - log10_dt_dzdz_dVc

    def mass_limit(self, log10_smbhb_dN: np.ndarray) -> float:
        """
        Function to determine the upper mass limit to integrate over
        Equation (9) of Sato-Polito & Kamionkowski (2023)
        """

        # integrate dN over z
        dN_dlogMdlogf = np.trapz(10**log10_smbhb_dN, axis=2, x=self.z)

        # integrate each frequency bin - convert from dlogf to df
        dN_dlogM = dN_dlogMdlogf * self.dlog10_f[:, None]

        mass_limit = np.zeros(self.Nfreqs)

        # integrate from max log10_M to limit to get unity
        for ii in range(self.Nfreqs):
            integral = cumulative_trapezoid(dN_dlogM[ii][::-1],
                                            x=self.log10_M)[::-1]
            for jj in range(len(integral)):
                if integral[jj] > 1 and integral[jj+1] < 1:
                    break
            mass_limit[ii] = self.log10_M[jj]

        return mass_limit

    def log10_h2(self, log10_M: float, z: float, log10_f: float) -> float:
        """
        log10 of the GW strain for a single SMBHB with a given chirp mass and
        redshift
        Equation (7) of Sato-Polito & Kamionkowski (2023)

        Parameters
        ----------
        log10_M : float
            log10 of the chirp mass
        z : float
            Redshift
        log10_f : float
            log10 of the observed frequency

        Returns
        -------
        log10_h2 : float
            log10 of the GW strain
        """

        # luminosity distance
        dL = cosmo.luminosity_distance(z).to('Mpc').value

        # GW strain
        log10_GM = np.log10(G) + log10_M
        log10_h2 = (np.log10(32 * np.pi**(4/3)) + (4/3)*log10_f +
                    (10/3)*log10_GM - np.log10(5 * c**8 * dL**2) +
                    (10/3)*np.log10(1+z))

        return log10_h2

    def log10_hc2_synth(self, model_params: dict) -> float:
        """
        log10 of the characteristic GW strain for the synthetic population

        Parameters
        ----------
        model_params : dict
            Model parameters

        Returns
        -------
        log10_hc2 : float
            log10 of the GW strain
        """

        # compute number of SMBHBs per log10 frequency,
        # chirp mass and redshift bin
        log10_N = np.zeros((self.Nfreqs, self.log10Mbins, self.zbins))

        for ii in range(self.Nfreqs):
            log10_M = self.log10_M[self.log10_M <= self.max_mass[ii]]
            for jj in range(len(log10_M)):
                for kk in range(self.zbins):
                    log10_N[ii, jj, kk] = self.smbhb_number_per_bin(
                        self.log10_f[ii], self.z[kk], log10_M[jj],
                        model_params
                    )

        # compute the characteristic strain squared
        log10_hc2 = np.zeros_like(self.log10_f)

        for ii in range(self.Nfreqs):
            log10_M = self.log10_M[self.log10_M <= self.max_mass[ii]]
            for jj in range(len(log10_M)):
                for kk in range(self.zbins):
                    # generate log10_M, z, log10_f for N SMBHBs
                    log10_M = np.random.uniform(self.log10_Mbin_edges[jj],
                                                self.log10_Mbin_edges[jj+1],
                                                int(10**log10_N[ii, jj, kk]))
                    z = np.random.uniform(self.zbin_edges[kk],
                                          self.zbin_edges[kk+1],
                                          int(10**log10_N[ii, jj, kk]))
                    log10_f = np.random.uniform(self.log10_fbin_edges[ii],
                                                self.log10_fbin_edges[ii+1],
                                                int(10**log10_N[ii, jj, kk]))

                    # compute the GW strain for each SMBHB
                    log10_h2 = self.log10_h2(log10_M, z, log10_f)
                    del log10_M, z  # save some memory
                    log10_h2f = log10_h2 + log10_f
                    del log10_f, log10_h2  # save some memory

                    # compute the characteristic strain squared
                    log10_h2f *= np.log(10)  # log10 to ln
                    log10_hc2[ii] += logsumexp(log10_h2f) / np.log(10)

        return log10_hc2 - np.log10(self.df)
