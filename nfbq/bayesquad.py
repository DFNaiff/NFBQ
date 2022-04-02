# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    Bayesian quadrature functions
"""

import torch


def monte_carlo_bayesian_quadrature(gp,
                                    nf,
                                    nsamples,
                                    return_var=True):
    """
        Bayesian quadrature using monte carlo sampling
        for estimating the associated kernel integrals

        Parameters
        ----------
        gp : SimpleGP
            The Gaussian process object to be integrated
        nf : NormalizingFlow
            The associated normalizing flow object
        nsamples : int
            The number of samples for monte carlo sampling
        return_var : bool
            If True, returns the mean and variance of the BQ integral
            If False, returns only mean

        Returns
        -------
        torch.Tensor,torch.Tensor (mean and var of BQ integral)
        or torch.Tensor, (mean of BQ integral) depending on return_var

    """
    samples1 = nf.sample(nsamples)
    z = gp.kernel_function(gp.X, samples1).mean(dim=-1, keepdim=True)  # (m,1)
    if return_var:
        samples2 = nf.sample(nsamples)
        samples3 = nf.sample(nsamples)
        gamma = gp.kernel_function(samples2, samples3, diagonal=True).mean()
    else:
        gamma = None
    mean, var = _calculate_bq_mean_var(gp, z, gamma)
    if var is None:
        return mean
    else:
        return mean, var


def monte_carlo_bayesian_elbo(gp,
                              nf,
                              nsamples,
                              return_var=True):
    """
        Estimation of ELBO between logprob approximated by GP
        and NF, by doing bayesian quadrature and MC integration 
        on the associated kernel integrals

        Parameters
        ----------
        gp : SimpleGP
            The Gaussian process object to be integrated
        nf : NormalizingFlow
            The associated normalizing flow object
        nsamples : int
            The number of samples for monte carlo sampling
        return_var : bool
            If True, returns the mean and variance of the BQ integral
            If False, returns only mean

        Returns
        -------
        torch.Tensor,torch.Tensor (mean and var of BQ integral)
        or torch.Tensor, (mean of BQ integral) depending on return_var

    """
    samples1, qdensity1 = nf.sample_and_logprob(nsamples)
    z = gp.kernel_function(gp.X, samples1).mean(dim=-1, keepdim=True)  # (m,1)
    if return_var:
        samples2 = nf.sample(nsamples)
        samples3 = nf.sample(nsamples)
        gamma = gp.kernel_function(samples2, samples3, diagonal=True).mean()
    else:
        gamma = None
    mean, var = _calculate_bq_mean_var(gp, z, gamma)
    mean += -qdensity1.mean()
    if var is None:
        return mean
    else:
        return mean, var
    

def _calculate_bq_mean_var(gp, z, gamma=None):
    """
        Calculate mean and variance of bayesian quadrature
        based on gp, zvector and gamma
    """
    y_, _ = torch.triangular_solve(gp.y_-gp.mean,
                                   gp.upper_chol_matrix,
                                   upper=True,
                                   transpose=True)  # (m,1)
    z_, _ = torch.triangular_solve(z,
                                   gp.upper_chol_matrix,
                                   upper=True,
                                   transpose=True)  # (m,1)
    mean = (gp.mean + z_.transpose(-2, -1)@y_)[0][0]  # (1,1) -> (,)
    if gamma is None:
        var = None
    else:
        var = (gamma - z_.transpose(-2, -1)@z_)[0][0]  # (1,1) -> (,)
    return mean+gp.ymax, var