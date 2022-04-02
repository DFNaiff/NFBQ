# -*- coding: utf-8 -*-
import torch

from . import activations
from . import implicit


class NormalizingLayer(torch.nn.Module):
    """
        Abstract class defining a normalizing layer

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
    """

    has_inverse = False

    def __init__(self, ndim):
        self.ndim = ndim
        super().__init__()

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
        raise NotImplementedError

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def logabsdet_autodiff(self, x):
        """
        Calculates log|J(f(x)| of the layer through automatic differentiation.
        Not-efficient, use only for sanity checks.

        Parameters
        ----------
        x : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The log|J(f(x)| value
        """
        jac = torch.autograd.functional.jacobian(self.forward, x)
        return torch.log(torch.abs(torch.det(jac)))


class ElementWiseLayer(NormalizingLayer):
    """
        Elementwise application of a 1d smooth function

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
    """
    has_inverse = True

    def __init__(self, ndim, h):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.
        h : Smooth1d
            Smooth element-wise inverse transformation class.
        """
        super().__init__(ndim)
        self.h = h

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
        return self.h.f(x)

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        return self.h.invf(y)

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
        return torch.sum(torch.log(torch.abs(self.h.df(x))), dim=-1)


class BaseTriuLayer(NormalizingLayer):
    """
        Transformation of the form
        f(x) = h(Wx + b), where h is a element-wise R -> R smooth (by parts) invertible
        function, and W is an upper triangular matrix, and b is a vector

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        h : Smooth1d
            Smooth element-wise inverse transformation class.
        wflat : torch.nn.Parameter
            Elements of the upper triangular part of W
        b : torch.nn.Parameter
            Dislocation vector
    """
    has_inverse = True

    def __init__(self, ndim, h):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.
        h : Smooth1d
            Smooth element-wise inverse transformation class.

        """
        super().__init__(ndim)
        self.h = h
        self.wflat = torch.nn.Parameter(torch.randn(ndim*(ndim+1)//2))
        self.b = torch.nn.Parameter(torch.zeros(ndim))

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
        y = x@self.W + self.b
        return self.h.f(y)

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        y0 = self.h.invf(y)
        x = torch.linalg.solve_triangular(self.W, y0-self.b, upper=True, left=False)
        return x

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
        W = self.W
        y = x@W + self.b
        jacdiag = self.h.df(y)*torch.diag(W)
        return torch.sum(torch.log(torch.abs(jacdiag)), dim=-1)

    @property
    def W(self):
        W = torch.zeros(self.ndim, self.ndim)
        inds = torch.tril_indices(self.ndim, self.ndim)
        W[inds[1], inds[0]] = self.wflat
        return W


class TriuLayer(NormalizingLayer):
    """
        Compound layer consisting of a permutation of the input joined
        followed by a BaseTriuLayer transformation

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        permlayer : RandomPermutationLayer
            Permutation layer
        baselayer : BaseTriuLayer
            Upper triangular transformation layer
    """
    has_inverse = True

    def __init__(self, ndim, h):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.
        h : Smooth1d
            Smooth element-wise inverse transformation class.

        """
        super().__init__(ndim)
        self.permlayer = RandomPermutationLayer(ndim)
        self.baselayer = BaseTriuLayer(ndim, h)

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
        return self.baselayer.forward(self.permlayer.forward(x))

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        return self.permlayer.inverse(self.baselayer.inverse(y))

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
        return self.baselayer.logabsdet(self.permlayer.forward(x))


class LeakyTriuLayer(TriuLayer):
    """
        TriuLayer with LeakyReLU as activation function
    """
    def __init__(self, ndim, a=0.01):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.
        a : float
            Leakyness of associated ReLU

        """
        super().__init__(ndim, activations.LeakyRelu(a))


class RandomPermutationLayer(NormalizingLayer):
    """
        Random permutation of the inputs

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        perm : torch.Tensor
            The permutation tensor
        invperm : torch.Tensor
            The inverse permutation tensor
    """
    has_inverse = True

    def __init__(self, ndim):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.

        """
        super().__init__(ndim)
        self.register_buffer("perm", torch.randperm(ndim))
        self.register_buffer("invperm", torch.argsort(self.perm))

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
        return x[..., self.perm]

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        return y[..., self.invperm]

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
        return torch.zeros_like(x)[0]


class TanhPlanarLayer(NormalizingLayer):
    """
        Radial flow layer as described in
        Rezende & Mohamed, 2015, https://arxiv.org/pdf/1505.05770.pdf

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        u : torch.nn.Parameter
            The u parameter
        beta : torch.nn.Parameter
            The w parameter
        x0 : torch.nn.Parameter
            The b parameter

    """
    has_inverse = True

    def __init__(self, ndim):
        """
        Attributes
        ----------
        ndim : int
            Dimension of the layer.

        """
        super().__init__(ndim)
        self.u = torch.nn.Parameter(torch.randn(ndim))
        self.w = torch.nn.Parameter(torch.randn(ndim))
        self.b = torch.nn.Parameter(torch.tensor(0.0))

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
        return x + self.uhat*torch.tanh((x@self.w).unsqueeze(-1) + self.b)

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        #w^y = alpha + w^T@u*h(alpha + b)
        #alpha : (...,)
        #z_paralel = alpha*w/||w||^2
        #z_perp = y - z - u*h(w^T@z + b)
        h = lambda x : torch.tanh(x)
        hprime = lambda x : 1 - torch.tanh(x)**2
        def f(alpha, wy, wu, b):
            return alpha + wu*h(alpha + b) - wy
        def df(alpha, wy, wu, b):
            return 1.0 + wu*hprime(alpha + b)
        uhat = self.uhat
        wu = (self.w*uhat).sum() #(,)
        wy = (y@self.w).unsqueeze(-1) #(..., 1)
        #TODO: Use instead implicit function theorem
        #and custom Function class
        alpha = implicit.newton_method(f, df,
                                       wy, wu, self.b) #(..., )
        wnorm2 = (self.w**2).sum()
        xpar = alpha*self.w/wnorm2 #(...,)
        xperp = y - xpar - uhat*h((xpar@self.w).unsqueeze(-1) + self.b)
        x = xpar + xperp
        return x

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
        hprime = lambda x : 1 - torch.tanh(x)**2
        psi = hprime((x@self.w).unsqueeze(-1) + self.b)*self.w  # (..., n)
        return torch.log(torch.abs(1 + psi@self.uhat))  # (...)

    @property
    def uhat(self):
        u = self.u
        w = self.w
        m = lambda x : -1.0 + torch.nn.functional.softplus(x)
        inner_product = (w*u).sum()
        wnorm2 = w.square().sum()
        return u + (m(inner_product) - inner_product)*w/(wnorm2)


class RadialLayer(NormalizingLayer):
    """
        Radial flow layer as described in
        Rezende & Mohamed, 2015, https://arxiv.org/pdf/1505.05770.pdf

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        raw_alpha : torch.nn.Parameter
            Untransformed alpha parameter
        beta : torch.nn.Parameter
            Beta scaling parameter
        x0 : torch.nn.Parameter
            Center parameter

    """
    has_inverse = True

    def __init__(self, ndim):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.

        """
        super().__init__(ndim)
        self.raw_alpha = torch.nn.Parameter(torch.randn(tuple()))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        self.x0 = torch.nn.Parameter(torch.randn(ndim))

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
        dx = x - self.x0
        r = torch.linalg.vector_norm(dx, dim=-1).unsqueeze(-1)  # (..., 1)
        h = 1/(self.alpha + r)  # (..., 1)
        return x + self.betahat*h*dx
    
    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        def f(r, alpha, beta, c):
            return r*(1 + beta/(alpha + r)) - c
        def df(r, alpha, beta, c):
            return beta/(r + alpha)*(1 - r/(r + alpha)) + 1.0
        alpha = self.alpha #(,)
        beta = self.betahat #(,)
        c = torch.linalg.vector_norm(y - self.x0, dim=-1).unsqueeze(-1) #(..., 1)
        r = implicit.newton_method(f, df, alpha, beta, c) #(...,)
        rxhat = (y - self.x0)/(1 + beta/(alpha + r))
        x = self.x0 + rxhat
        return x
    
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
        betahat = self.betahat
        dx = x - self.x0
        r = torch.linalg.vector_norm(dx, dim=-1).unsqueeze(-1)  # (..., 1)
        h = 1/(self.alpha + r)  # (..., )
        hprime = -1/((self.alpha+r)**2)
        det = (1 + betahat*h)**(self.ndim-1)*(1 + betahat*h + betahat*hprime*r)
        return torch.log(torch.abs(det))

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha)

    @property
    def betahat(self):
        return -self.alpha + torch.nn.functional.softplus(self.beta)


class AffineLayer(NormalizingLayer):
    """
        Basic scale transformation, so that
        f(x) = Ax + b, with A being a diagonal matrix

        Attributes
        ----------
        ndim : int
            Dimension of the transformation
        has_inverse : bool
            Whether the layer has an implemented inverse function
        a : torch.nn.Parameter
            Diagonal of A
        b : torch.nn.Parameter
            Tensor of the dislocation vector
    """
    has_inverse = True

    def __init__(self, ndim):
        """
        Parameters
        ----------
        ndim : int
            Dimension of the layer.

        """
        super().__init__(ndim)
        self.a = torch.nn.Parameter(torch.ones(ndim))
        self.b = torch.nn.Parameter(torch.zeros(ndim))

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
        return self.a*x + self.b

    def inverse(self, y):
        """
        Calculates f^{-1}(y)

        Parameters
        ----------
        y : torch.Tensor
            The input where value will be calculated

        Returns
        -------
        torch.Tensor
            The f^{-1}(x) value
        """
        return (y - self.b)/self.a

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
        return torch.sum(torch.log(torch.abs(self.a)))
