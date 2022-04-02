# -*- coding: utf-8 -*-

import torch

from . import activations
from . import layers


class WarpingLayer(layers.NormalizingLayer):
    """
        Warping layer of transformation
        
        Attributes
        ----------
        ndim : int
            Dimension.
        activation_map : Dict[Union[torch.Tensor, List], Smooth1d]
            Map for each index
    """
    
    def __init__(self, ndim, activation_map):
        """
        Parameters
        ----------
        ndim : int
            Dimension.
        activation_map : Dict[torch.Tensor, Smooth1d]
            Map for each index
        """
        super().__init__(ndim)
        self._make_submaps(activation_map)
        #self._assert_validity_activation_map()
        

    def forward(self, x):
        """
        Calculates f(x)
        
        Parameters
        ----------
        x : torch.Tensor
            The input where value will be calculated
        
        Returns
        -------
        torch.Tensor
            The f(x) value
        """
        y = self._gather(x, 'f')
        return y
    
    def logabsdet(self, x):
        """
        Calculates log|J(f(x)| of the layer.
        
        Parameters
        ----------
        x : torch.Tensor
            The input where value will be calculated
        
        Returns
        -------
        torch.Tensor
            The log|J(f(x)| value
        """
        y = self._gather(x, 'df')
        res = torch.sum(torch.log(torch.abs(y)), dim=-1)
        return res
    
    def _gather(self, x, method):
        indexes = []
        vals = []
        for name, activation in self._activation_name_map.items():
            ind = self._index_map[name]
            val = getattr(activation, method)(x[..., ind])
            indexes.append(ind)
            vals.append(val)
        iperm = _make_iperm(*indexes)
        y = torch.cat(vals, dim=-1)[..., iperm]
        return y
    
    def _make_submaps(self, activation_map):
        fmap = torch.nn.ModuleDict()
        imap = dict()
        for i, (index, activation) in enumerate(activation_map.items()):
            name = 'subwarp%i'%i
            index = torch.tensor(index, dtype=torch.long) if \
                    not isinstance(index, torch.Tensor) else \
                    index
            imap[name] = index
            fmap[name] = activation
        self._activation_name_map = fmap
        self._index_map = imap

    def _assert_validity_activation_map(self, activation_map):
        raise NotImplementedError
        try:
            holder = []
            for key, value in activation_map.items():
                assert(key.ndim == 1)
                assert(isinstance(value, activations.Smooth1d))
                holder.append(key)
            assert(torch.all(torch.sort(torch.cat(holder, axis=-1))[0] == \
                             torch.arange(self.ndim, dtype=torch.long)))
        except:
            raise ValueError("Variable activation_map is invalid")


def make_warping_from_tuples(ndim, tuple_map, scales_map=None):
    """
    Makes the warping whose indexes are warped to the function defined by
    tuple_to_activation(tuple_map[ind], scales_map[ind]). See the documentation
    of that function for more information.

    Parameters
    ----------
    ndim : int
        Dimension of the warping.
    tuple_map : Dict[Union[List, torch.Tensor], List[Optional[float]]]
        Map from indices to the tuple bounds
    scales_map : Optional[Dict[Union[List, torch.Tensor], List[Optional[float]]]], optional
        Map from indices to scales. If None, assumed to be empty dict.
        If there is no corresponding item, is assumed that scales_map[indices] = None.
        The default is None.

    Returns
    -------
    WarpingLayer
        The corresponding warping

    """
    if scales_map is None:
        scales_map = dict()
    activation_map = dict()
    for key, value in tuple_map.items():
        scale = scales_map.get(key, None)
        new_key = torch.tensor(key, dtype=torch.long)
        activation_map[new_key] = tuple_to_activation(value, scale)
    return WarpingLayer(ndim, activation_map)
    
    
def make_default_positive_warping(ndim):
    """
    Makes the warping from R^n to R^{+, n} such that f_i(x_i) = softplus(x_i)
    
    Parameters
    ----------
    ndim : int
        The dimension of the map
        
    Returns
    -------
    WarpingLayer
        The corresponding warping
    """
    index = torch.arange(ndim, dtype=torch.long)
    activation = activations.Softplus()
    activation_map = {index: activation}
    return WarpingLayer(ndim, activation_map)

    
def make_default_box_warping(ndim):
    """
    Makes the warping from R^n to (0, 1)^n, such that f_i(x_i) = sigmoid(x_i)
    
    Parameters
    ----------
    ndim : int
        The dimension of the map
        
    Returns
    -------
    WarpingLayer
        The corresponding warping
    """
    index = torch.arange(ndim, dtype=torch.long)
    activation = activations.Sigmoid()
    activation_map = {index: activation}
    return WarpingLayer(ndim, activation_map)


def tuple_to_activation(t, scale=None):
    """
    Makes map from R -> (lb, ub), scaled if necessary.
    Here, lb = t[0] is a lower bound. In case lb is None,
    it is assumed to be unbounded. Same thing for ub = t[1].
    The corresponding maps are:
        (None, None) -> f(x) = xs
        (lb, None) -> f(x) = softplus(xs) + lb
        (None, ub) -> f(x) = -softplus(-xs) + ub
        (lb, ub) -> f(x) = (ub - lb)*sigmoid(xs) + lb
    where xs = x/scale if scale is a float, otherwise xs = x
    
    Parameters
    ----------
    t : Tuple[Optional[float]]
    scale : Optional[float]
    
    Returns
    -------
    Smooth1d
        The corresponding activation
    """
    lb = t[0]
    ub = t[1]
    if lb is None and ub is None:
        activation = activations.Affine()
    elif lb is not None and ub is None:
        activation = activations.Affine(b=lb)@activations.Softplus()
    elif lb is None and ub is not None:
        activation = activations.Affine(a=-1.0, b=ub)@activations.Softplus()@activations.Affine(a=-1.0)
    else:
        assert lb < ub
        activation = activations.Affine(a=ub-lb, b=lb)@activations.Sigmoid()
    if scale is not None:
        scale = float(scale)
        activation = activation@activations.Affine(a=1.0/scale)
    return activation


def _index_split(indexes, ndim):
    return torch.split(torch.arange(ndim, dtype=torch.long), indexes)


def _make_perm(*inds):
    return torch.cat([i for i in inds], axis=-1)


def _make_iperm(*inds):
    return torch.argsort(_make_perm(*inds))