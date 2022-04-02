# -*- coding: utf-8 -*-
import math

import torch

from .. import utils


class BaseDistrib(torch.nn.Module):
    """
        Base distribution class. Use one of the derived classes
    """
    def __init__(self, ndim):
        self.ndim = ndim
        super().__init__()
        
    def logprob(self, x):
        pass
    
    def sample(self, nsamples):
        pass


class NormalDistrib(BaseDistrib):
    """
        Multivariate normal distribution with diagonal covariance
        
        Parameters
        ----------
        mean : torch.Tensor
            Mean
        scale : torch.Tensor
            Standard deviation
    """
    def __init__(self, ndim, mean=0.0, scale=1.0):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the distribution.
        mean : Union[float, torch.Tensor, List], optional
            Mean of the distribution. The default is 0.0.
        scale : Union[float, torch.Tensor, List], optional
            Standar deviation of the distribution. The default is 1.0.
        """
        super().__init__(ndim)
        self.register_buffer("mean", utils.tensor_convert(mean))
        self.register_buffer("scale", utils.tensor_convert(scale))
    
    def logprob(self, x):
        """
        Log-density
        
        Parameters
        ----------
        x : torch.Tensor
            Input value (at the flown domain)

        Returns
        -------
        torch.Tensor
            Associated log-density

        """
        #(..., n) -> (...)
        assert(x.shape[-1] == self.ndim)
        exp_term = -0.5*((x - self.mean)/self.scale).square().sum(dim=-1) #(...)
        normalizing_term = -torch.sum(torch.log(self.scale)) - self.ndim/2*math.log(2*math.pi)
        res = exp_term + normalizing_term
        return res
    
    def sample(self, nsamples):
        """
        Sample from distribution
        
        Parameters
        ----------
        nsamples : int
            Number of samples
            
        Returns
        -------
        torch.Tensor
            Samples of distribution
        """
        return torch.randn(nsamples, self.ndim)*self.scale + self.mean