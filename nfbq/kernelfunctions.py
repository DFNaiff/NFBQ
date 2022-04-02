# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    Kernel functions for GP
"""

import math
import torch


_SEPARABLE_KERNELS = ['smatern12', 'smatern32', 'smatern52', 'sqe']


def kernel_function(x1, x2, theta=1.0, l=1.0, kind='sqe',
                    output='pairwise'):
    """
    Returns kernel values

    Parameters
    ----------
    x1 : torch.Tensor
        First kernel argument
    x2 : torch.Tensor
        Second kernel matrix
    theta : float or torch.Tensor
        Output scale
    l : float or torch.Tensor
        Lengthscale
    kind : str
        Kind of kernel. Must be in
        ['sqe','matern12','matern32','matern52',
         'smatern12','smatern32','smatern52']
    output : str
        If 'pairwise', return pairwise kernel evaluations tensor
        If 'diagonal', return diagonal of pairwise kernel evaluations tensor

    Returns
    -------
    torch.Tensor
        Kernel values
    """
    # if output == 'pairwise'
    #x1 : (...,d)
    # x2 : (...*,d)
    # return : (...,...*)
    # elif output == 'diagonal'
    #x1 : (...,d)
    #x2 : (...,d)
    # return : (...,)
    if output == 'pairwise':
        x1_ = x1[(slice(None),)*(x1.ndim-1) + (None,)*(x2.ndim-1)]
        difference = x1_ - x2
    elif output == 'diagonal':
        difference = x1 - x2
    else:
        raise ValueError
    if kind in ['sqe', 'matern12', 'matern32', 'matern52']:
        r = torch.linalg.vector_norm(difference/l, dim=-1)
        if kind == 'sqe':
            res = theta*sqe(r)
        if kind == 'matern12':
            res = theta*matern12(r)
        elif kind == 'matern32':
            res = theta*matern32(r)
        elif kind == 'matern52':
            res = theta*matern52(r)
    elif kind in ['smatern12', 'smatern32', 'smatern52']:
        r = torch.abs(difference/l)
        if kind == 'smatern12':
            res = theta*torch.prod(matern12(r), dim=-1)
        elif kind == 'smatern32':
            res = theta*torch.prod(matern32(r), dim=-1)
        elif kind == 'smatern52':
            res = theta*torch.prod(matern52(r), dim=-1)
    else:
        raise NotImplementedError
    return res


def kernel_function_separated(x1, x2, theta=1.0, l=1.0, kind='sqe',
                              output='pairwise'):
    """
    Return kernel values for each component of the values,
    usable only for separable kernels

    Parameters
    ----------
    x1 : torch.Tensor
        First kernel argument
    x2 : torch.Tensor
        Second kernel matrix
    theta : float or torch.Tensor
        Output scale
    l : float or torch.Tensor
        Lengthscale
    kind : str
        Kind of kernel. Must be in
        ['sqe','smatern12','smatern32','smatern52']
    output : str
        If 'pairwise', return pairwise kernel evaluations tensor
        If 'diagonal', return diagonal of pairwise kernel evaluations tensor

    Returns
    -------
    torch.Tensor
        Kernel values
    """
    #x1 : (...,d)
    # x2 : (...*,d)
    # return : (...,...*,d) or (...,d)
    assert kind in _SEPARABLE_KERNELS
    if output == 'pairwise':  # (...,...*,d)
        x1_ = x1[(slice(None),)*(x1.ndim-1) + (None,)*(x2.ndim-1)]
        difference = x1_ - x2
    elif output == 'diagonal':
        difference = x1 - x2  # (...,d)
    else:
        raise ValueError
    r = torch.abs(difference/l)
    d = r.shape[-1]
    if kind == 'sqe':
        res = theta**(1.0/d)*sqe(r)
    elif kind == 'smatern12':
        res = theta**(1.0/d)*matern12(r)
    elif kind == 'smatern32':
        res = theta**(1.0/d)*matern32(r)
    elif kind == 'smatern52':
        res = theta**(1.0/d)*matern52(r)
    return res


def sqe(r):
    """SQE function"""
    return torch.exp(-0.5*r**2)


def matern12(r):
    """Matern 1/2 function"""
    return torch.exp(-r)


def matern32(r):
    """Matern 3/2 function"""
    return (1+math.sqrt(3)*r)*torch.exp(-math.sqrt(3)*r)


def matern52(r):
    """Matern 5/2 function"""
    return (1+math.sqrt(5)*r+5./3*r**2)*torch.exp(-math.sqrt(5)*r)
