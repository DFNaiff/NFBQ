# -*- coding: utf-8 -*-
import collections
import copy

import torch

from .. import utils
from . import warpinglayers


class NamedFlow(object):
    """
    A distribution wrapper for distributions of named random 
    variables in product subsets of R^n, with each variable 
    being either defined on the real line, bounded from below, 
    or bounded by an interval.

    Attributes
    ---------
    basedistrib : distributions.ProbabilityDistribution
        Base distribution in R^n
    param_dict : dict
        Dictionary with keys containing parameter names,
        and values being a dict consisting of
        dim : int
            Dimension of parameter    
        bound : (float,float)
            Bounds of parameter
        scale : torch.Tensor
            Scaling factor of parameter
    """
    def __init__(self,
                 params_name,
                 params_dim,
                 params_bound,
                 params_scale=None):
        """
        Parameters
        ----------
        params_name : List[str]
            Name of parameters
        params_dim : List[int]
            Dimension of each parameter in params_name
        params_bound : List[(float,float)]
            Lower and upper bound for each parameter in params_name
        params_scale : None or dict[str,List[float]} dict
            If not None, the scale factor of each parameter

        """
        self._set_param_dict(params_name, params_dim, params_bound, params_scale)
        
    def set_flow(self, baseflow, makecopy=True):
        """
        Sets the base flow and appends it to the warping layer.
        Note that by default THIS MODIFIES THE BASE FLOW NET.
        If you want for it to not be modified, set copy=True
        
        Parameters
        ----------
        baseflow : NormalizingFlow
        copy : bool
        """
        assert(baseflow.ndim == self.total_dim)
        if makecopy:
            baseflow = copy.deepcopy(baseflow)
        warplayer = self._make_warp_layer()
        self.nflow = baseflow
        self.nflow.flows.append(warplayer)
        
    def logprob(self, params):
        """
        Log probability

        Parameters
        ----------
        params : dict[str,List[float]]
            The parameter values to be calculated log density
        numpy : bool
            If False, return torch.Tensor
            If True, return np.array

        Returns
        -------
        torch.Tensor or np.array
            Values of log density

        """
        x = self.join_parameters(params)
        return self.nflow.logprob(x)
    
    def sample(self, n):
        """
        Sample

        Parameters
        ----------
        n : int
            Number of samples
        numpy : bool
            If False, return torch.Tensor as values
            If True, return np.array as values
        Returns
        -------
        params : dict[str,torch.Tensor]
            Samples from distribution

        """
        if not hasattr(self, "nflow"):
            raise ValueError("Normalizing flow not set")
        x = self.nflow.sample(n)
        xdict = self.split_parameters(x)
        return xdict

    def flow_forward(self, params):
        """
        Flow forward the input
        
        Parameters
        ----------
        x : dict[str,torch.Tensor]
            Value at base of the flow.

        Returns
        -------
        torch.Tensor
            Transformed flow.

        """
        x = self.join_parameters(params)
        x = self.nflow.flow_forward(x)
        xdict = self.split_parameters(x)
        return xdict
    
    def ulogprob(self, params):
        """
        Flow forward the input and calculate density at end
        
        Parameters
        ----------
        x : dict[str,torch.Tensor]
            Value at base of the flow.

        Returns
        -------
        dict[str,torch.Tensor], torch.Tensor
            Value at the end of the flow and associated log-density

        """
        x = self.join_parameters(params)
        x, y = self.nflow.ulogprob(x)
        xdict = self.split_parameters(x)
        return xdict, y

    def sample_and_logprob(self, nsamples):
        """
        Sample from normalizing flow and return 
        the log-density at these samples
        
        Parameters
        ----------
        nsamples : int
            Number of samples
            
        Returns
        -------
        dict[str,torch.Tensor], torch.Tensor
            Samples of flow distribution and log-densities
        """
        
        x = self.nflow.sample(nsamples)
        x, y = self.nflow.ulogprob(x)
        xdict = self.split_parameters(x)
        return xdict, y
    
    def join_parameters(self, params):
        """
        Join the parameters in a tensor

        Parameters
        ----------
        params : dict[str,torch.Tensor]
            The parameter values to be joined

        Returns
        -------
        torch.Tensor
            The joint parameter matrix
        """

        res = torch.cat([params[name] for name in self.names], dim=-1)
        return res

    def split_parameters(self, x):
        """
        Split the tensor x into each corresponding parameter

        Parameters
        ----------
        torch.Tensor
            The joint parameter tensor

        Returns
        -------
        params : dict[str,torch.Tensor]
            The splitted parameter values

        """

        splits = torch.split(x, [self.dim(name) for name in self.names], dim=-1)
        params = {name: splits[i] for i, name in enumerate(self.names)}
        return params
    
    def dim(self, key):
        """int : returns the dimension of 'key' parameter"""
        return self.param_dict[key]['dim']

    def bound(self, key):
        """tuple[float] : Returns the bound of 'key' parameter"""
        return self.param_dict[key]['bound']

    def scale(self, key):
        """tuple[float] : Returns the scale of 'key' parameter"""
        return self.param_dict[key]['scale']
    
    @property
    def has_logprob(self):
        return self.nflow.has_logprob
    
    @property
    def names(self):
        """list[str] : names of parameters"""
        return list(self.param_dict.keys())

    @property
    def dims(self):
        """dict[str,int] : dimension dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'dim')

    @property
    def bounds(self):
        """dict[str:(float,float)] : sounds dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'bound')

    @property
    def scales(self):
        """dict[str:List[float]] : scales dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'scale')

    @property
    def total_dim(self):
        """int : total dimension of underlying domain"""
        return sum(self.dims.values())
    
    def _set_param_dict(self, params_name, params_dim, params_bound, params_scale):
        if params_scale is None:
            params_scale = dict()
        param_dict = collections.OrderedDict()
        for _, (name, dim, bound) in enumerate(zip(params_name, params_dim, params_bound)):
            lb, ub = bound
            scale = utils.tensor_convert(params_scale.get(name, 1.0))
            param_dict[name] = {'dim': dim, 'bound': bound, 'scale': scale}
        self.param_dict = param_dict
        
    def _make_warp_layer(self):
        dims = []
        bounds = []
        scales = []
        for _, value in self.param_dict.items():
            dims.append(value['dim'])
            bounds.append(value['bound'])
            scales.append(value['scale'])
        warplayer = warpinglayers.make_warping_from_slice(dims, bounds, scales)
        return warplayer
