# -*- coding: utf-8 -*-

"""
    A collection of acquisition functions to be used

    All functions are of format f(q(x),mean(x),var(x))
    where q(x) is the current distribution of BVBQ,
    mean(x) and var(x) are the mean and variances
    of the associated Gaussian Process at x
"""

import torch



def acqfunction(x, gp, nf, acqrule):
    """
    Prospective prediction acquisition function
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation point
    gp : gp.SimpleGP
        GP approximating (warped) log density
    nf : NormalizingFlow
        Normalizing flow
    acqrule : Callable[(torch.Tensor, torch.Tensor, torch.Tensor), torch.Tensor]
        The acquisition rule
    Returns
    -------
    torch.Tensor
        The acquisition function value
    """
    xforward, logprob = nf.ulogprob(x)
    mean, var = gp.predict(xforward, to_return='var')
    return acqrule(logprob, mean, var)


def acqwrapper(acqrule):
    def f(x, gp, namedflow):
        return acqfunction(x, gp, namedflow, acqrule)
    return f


@acqwrapper
def prospective_prediction(mean, var, logprob):
    res = torch.exp(mean+2*logprob)*var
    return res

@acqwrapper
def moment_matched_log_transform(mean, var, logprob):
    res = torch.exp(2*mean + var)*(torch.exp(var)-1)
    return res

@acqwrapper
def warped_entropy(mean, var, logprob):
    res = torch.log(var)/2.0 + mean
    return res

@acqwrapper
def prospective_mmlt(mean, var, logprob):
    res = torch.exp(2*mean+2*logprob+var)*(torch.exp(var)-1)
    return res
