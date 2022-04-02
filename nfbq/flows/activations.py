# -*- coding: utf-8 -*-

import torch


class Smooth1d(torch.nn.Module):
    """
        Abstract class for a smooth invertible element wise activation
    """
    def __init__(self):
        super().__init__()
    
    def f(self, x : torch.Tensor) -> torch.Tensor:
        """
        Function evaluation
        """
        raise NotImplementedError
    
    def df(self, x : torch.Tensor) -> torch.Tensor:
        """
        Derivative evaluation
        """
        raise NotImplementedError
    
    def invf(self, y : torch.Tensor) -> torch.Tensor:
        """
        Inverse evaluation
        """
        raise NotImplementedError
    
    def __matmul__(self, other):
        return Composition(self, other)


class LeakyRelu(Smooth1d):
    """
        Leaky ReLU activation, R -> R
        
        Attributes
        ----------
        a : torch.Tensor
            The "leakyness" of the ReLU
    """
    def __init__(self, a=0.1):
        """
        Parameters
        ----------
        a : Union[float, torch.Tensor]
            The "leakyness" of the ReLU
        """
        super().__init__()
        self.register_buffer('a', torch.tensor(a))
    
    def f(self, x : torch.Tensor) -> torch.Tensor:
        """
        Function evaluation
        """
        return x*(x>=0) + self.a*x*(x < 0)
    
    def df(self, x : torch.Tensor) -> torch.Tensor:
        """
        Derivative evaluation
        """
        return 1.0*(x>=0) + self.a*(x<0)
    
    def invf(self, y : torch.Tensor) -> torch.Tensor:
        """
        Inverse evaluation
        """
        return y*(y>=0) + 1.0/self.a*y*(y<0)


class Softplus(Smooth1d):
    """
        f(x) = log(1 + exp(x)), R -> R^+
    """
    def f(self, x):
        return torch.nn.functional.softplus(x)
    
    def df(self, x):
        return torch.sigmoid(x)
    
    def invf(self, x):
        return torch.where(x < 20.0,
                           torch.log(torch.exp(x) - 1),
                           x)
                           

class Affine(Smooth1d):
    """
        f(x) = ax + b, R -> R
    """
    def __init__(self, a=1.0, b=0.0):
        super().__init__()
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('b', torch.tensor(b))
        
    def f(self,x : torch.Tensor) -> torch.Tensor:
        return self.a*x + self.b
    
    def df(self, x : torch.Tensor) -> torch.Tensor:
        return self.a*torch.ones_like(x)

    def invf(self, y : torch.Tensor) -> torch.Tensor:
        return (y - self.b)/self.a


class Sigmoid(Smooth1d):
    """
        f(x) = 1/(1 + exp(-x)), R -> (0, 1)
    """
    def f(self, x : torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    def g(self, x : torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.sigmoid(x))*torch.sigmoid(x)
    
    def invf(self, y : torch.Tensor) -> torch.Tensor:
        return torch.logit(y)


class Composition(Smooth1d):
    """
        f(x) = g(h(x)), domain(h) -> image(g)
    """
    def __init__(self, g, h):
        super().__init__()
        self.g = g
        self.h = h
    
    def f(self, x : torch.Tensor) -> torch.Tensor:
        return self.g.f(self.h.f(x))
    
    def df(self, x : torch.Tensor) -> torch.Tensor:
        return self.g.df(self.h.f(x))*self.h.df(x)
    
    def invf(self, y : torch.Tensor) -> torch.Tensor:
        return self.h.invf(self.g.invf(y))
    

class Inverse(Smooth1d):
    """
        f(x) = g^{-1}(x), image(g) -> domain(g)
    """
    def __init__(self, g):
        super().__init__()
        self.g = g
    
    def f(self, x : torch.Tensor) -> torch.Tensor:
        return self.g.invf(x)
    
    def df(self, x : torch.Tensor) -> torch.Tensor:
        return 1.0/self.g.df(self.g.invf(x))
    
    def invf(self, y : torch.Tensor) -> torch.Tensor:
        return self.g.f(y)