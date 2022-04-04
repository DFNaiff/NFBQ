# -*- coding: utf-8 -*-
# pylint: disable=E1101

import collections
import math

import numpy as np
import torch
import dict_minimize.torch_api

from . import kernelfunctions
from . import utils


class SimpleGP(object):
    """
    Gaussian Process class for surrogate modeling

    Attributes
    ----------
    dim : int
        Dimension of the GP
    kind : str
        Kernel for GP
    min_jitter : float
        Minimum value for jittering kernel matrix
    fixed_params : set
        Name of fixed params when optimizing
    zeromax : bool
        Whether zeromax normalization is used
    ymax : float
        If zeromax is True, maximum of output data. Otherwise 0
    ndata : int
        Number of data points in GP
    X : torch.Tensor
        Data matrix
    y_ : torch.Tensor
        Evaluation matrix after zeromax normalization
    kernel_matrix : torch.Tensor
        Kernel matrix of data
    upper_chol_matrix : torch.Tensor
        Upper Cholesky factor of data's kernel matrix
    """

    def __init__(self, dim,
                 kind='sqe',
                 theta=1.0,
                 lengthscale=1.0,
                 noise=1e-2,
                 mean=0.0,
                 ard=True,
                 min_jitter=1e-4,
                 fixed_params=tuple(),
                 zeromax=False):
        """
        Parameters
        ----------
        dim : int
            Dimension of the GP
        lengthscale : float or [float]
            Kernel lengthscale
        theta : float
            Kernel outputscale
        noise : float
            Noise of GP
        mean : float
            Mean of GP
        kind : str
            Kernel for GP
        ard : bool
            Whether kernel has a lengthscale for every dimension
        min_jitter : float
            Minimum value for jittering kernel matrix
        fixed_params : set
            Name of fixed params when optimizing
        zeromax : bool
            Whether zeromax normalization is used
        """
        self.dim = dim
        self.kind = kind
        self.min_jitter = min_jitter
        self.fixed_params = set(fixed_params)
        self.zeromax = zeromax
        self.ymax = 0.0  # Neutral element in sum
        self.ndata = 0
        self.X = None
        self.y_ = None
        self.kernel_matrix = None
        self.upper_chol_matrix = None
        self.mean = mean  # THIS IS THE MEAN AFTER ZEROMAX TRANSFORMATION.
        # IF NOT USING ZEROMAX TRANSFORMATION, IGNORE
        # THIS WARNING
        self.theta = theta
        self.lengthscale = self._set_lengthscale(lengthscale, ard)
        self.noise = noise

    def set_data(self, X, y, empirical_params=False):
        """
        Set data for GP

        Parameters
        ----------
        X : List[float,float]
            Data matrix for GP
        y : List[float]
            Evaluation vector for GP
        empirical_params : bool
            Whether kernel parameters are initialized empirically
        """
        ndata = X.shape[0]
        X, y = utils.tensor_convert_(X, y)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, -1)  # (d,1)
        assert y.shape[0] == ndata
        assert y.shape[1] == 1
        assert X.shape[1] == self.dim
        if self.zeromax:
            self.ymax = torch.max(y)
        y_ = y - self.ymax  # Output after output rescaling
        if empirical_params:
            self.set_params_empirical_(X, y)
        kernel_matrix_ = self._make_kernel_matrix(
            X, X, self.theta, self.lengthscale)
        kernel_matrix = self._noisify_kernel_matrix(kernel_matrix_, self.noise)
        upper_chol_matrix = self._make_cholesky(kernel_matrix)
        self.ndata = ndata
        self.X = X
        self.y_ = y_
        self.kernel_matrix = kernel_matrix  # Noised already
        self.upper_chol_matrix = upper_chol_matrix

    def predict(self, xpred, to_return='mean'):
        """
        Gaussian process prediction

        Parameters
        ----------
        xpred : List[float]
            Prediction points
        to_return : str
            If 'mean', return only mean of prediction
            If 'var', return mean and variance of prediction
            If 'cov' and xpred has less than 3 dimensions,
            return mean and covariance matrix of prediction
        
        Returns
        -------
        torch.Tensor or torch.Tensor, torch.Tensor
            Mean, mean and varince, or mean and covariance, 
            depending on to_return
        """
        xpred = utils.tensor_convert(xpred)
        s = xpred.shape
        d = len(s)
        if d == 1:
            xpred = torch.unsqueeze(xpred, 0)
            # No difference for one item
            res = self._predict(xpred, to_return=to_return)
            if to_return == "mean":
                return res.squeeze(0)
            else:
                return (res[0].squeeze(0), res[1].squeeze(0))
        elif d == 2:
            res = self._predict(xpred, to_return=to_return)
        else:  # some reshaping trick in order
            # [...,d] -> [n,d]
            print(
                "If tensor has more than 2 dimensions, only diagonal of covariance is returned")
            if to_return == 'cov':
                to_return = 'var'
            n = int(np.prod((s[:-1])))
            xpred_r = xpred.reshape(n, s[-1])
            res_r = self._predict(
                xpred_r, to_return=to_return)
            if to_return == 'mean':
                mean_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                res = mean
            else:
                mean_r, var_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                var = var_r.reshape(*(s[:-1]))
                res = mean, var
        if to_return == 'mean':
            mean = res
            return mean
        else:
            mean, cov = res
            return mean, cov

    def loo_mean_prediction(self, xpred):  # Only return mean
        """Prediction using LOO. Not implemented"""
        raise NotImplementedError

    def optimize_params_qn(self, fixed_params=tuple(),
                           method='L-BFGS-B',
                           tol=1e-1, options=None):
        """
        Optimize the GP parameters via dict-minimize (Quasi-Newton method)

        Parameters
        ----------
        fixed_params : List[str]
            Parameters to be fixed in this optimization
        method : str
            Optimization method for dict-minimize
        tol : float
            Optimization tolerance for dict-minimize
        options : dict or None
            Options for dict-minimize

        Returns
        -------
        dict[str,torch.Tensor]
            A copy of optimal values found
        """

        params = {'raw_theta': self._raw_theta,
                  'mean': self.mean,
                  'raw_lengthscale': self._raw_lengthscale,
                  'raw_noise': self._raw_noise}
        params = utils.crop_fixed_params_gp(params,
                                            self.fixed_params.union(fixed_params))
        params = collections.OrderedDict(params)
        dwrapper = utils.dict_minimize_torch_wrapper(
            self._loglikelihood_wrapper)
        options = dict() if options is None else options
        res = dict_minimize.torch_api.minimize(dwrapper,
                                               params,
                                               method=method,
                                               tol=tol,
                                               options=options)
        res = {key: value.detach().clone() for key, value in res.items()}
        self.theta = self._derawfy(res.get('raw_theta', self._raw_theta))
        self.noise = self._derawfy(res.get('raw_noise', self._raw_noise))
        self.mean = res.get('mean', self.mean)
        self.lengthscale = self._derawfy(
            res.get('raw_lengthscale', self._raw_lengthscale))
        self.set_data(self.X, self.y)
        return res

    def optimize_params_sgd(self, fixed_params=tuple(),
                            maxiter=100,
                            optim=torch.optim.Adam,
                            lr=1e-1,
                            verbose=False):
        """
        Optimize the GP parameters via torch.optim (SGD method)

        Parameters
        ----------
        fixed_params : List[str]
            Parameters to be fixed in this optimization
        maxiter : int
            Number of steps for SGD
        optim : torch.optim.Optimizer
            Optimizer to be used
        lr : float
            Learning rate of optimizer
        verbose : bool
            Whether to show maxiter steps

        Returns
        -------
        dict[str,torch.Tensor]
            A copy of optimal values found
        """

        params = {'raw_theta': self._raw_theta,
                  'mean': self.mean,
                  'raw_lengthscale': self._raw_lengthscale,
                  'raw_noise': self._raw_noise}
        params = utils.crop_fixed_params_gp(params,
                                            self.fixed_params.union(fixed_params))
        for _, tensor in params.items():
            tensor.requires_grad = True
        optimizer = optim(list(params.values()), lr=lr)
        for i in range(maxiter):
            optimizer.zero_grad()
            loss = self._loglikelihood_wrapper(params)
            loss.backward()
            if verbose:
                print(i, loss.detach().numpy().item())
                print([(p, v.detach().numpy()) for p, v in params.items()])
                print('-'*5)
            optimizer.step()
        res = {key: value.detach().clone() for key, value in params.items()}
        self.theta = self._derawfy(res.get('raw_theta', self._raw_theta))
        self.noise = self._derawfy(res.get('raw_noise', self._raw_noise))
        self.mean = res.get('mean', self.mean)
        self.lengthscale = self._derawfy(
            res.get('raw_lengthscale', self._raw_lengthscale))
        self.set_data(self.X, self.y)
        return res

    def gradient_step_params(self, fixed_params=tuple(),
                             alpha=1e-1, niter=1):
        """Not implemented"""
        raise NotImplementedError

    def current_loglikelihood(self):
        """
            Get loglikelihood of GP

            Returns
            -------
            torch.Tensor
                Current loglikelihood
        """
        return self._loglikelihood(self.theta, self.lengthscale, self.noise, self.mean)

    def update(self, Xnew, ynew):
        """
            Update data

            Parameters
            ----------
            Xnew : List[float]
                Data matrix of update points
            ynew : List[float]
                Evaluation vector of update points
        """
        # FIXME : Broken somehow
        raise NotImplementedError
        nnew = Xnew.shape[0]
        Xnew, ynew = utils.tensor_convert_(Xnew, ynew)
        Xnew = torch.atleast_2d(Xnew)
        ynew = torch.atleast_2d(ynew)
        if self.zeromax:
            self.ymax = max(torch.max(ynew), self.ymax)
        assert ynew.shape[0] == nnew
        assert ynew.shape[1] == 1
        assert Xnew.shape[1] == self.dim
        Xup = torch.vstack([self.X, Xnew])
        yup = torch.vstack([self.y, ynew])
        K11 = self.kernel_matrix
        K12 = self._make_kernel_matrix(self.X, Xnew,
                                       self.theta,
                                       self.lengthscale)
        K21 = K12.transpose(-2, -1)
        K22_ = self._make_kernel_matrix(Xnew, Xnew,
                                        self.theta,
                                        self.lengthscale)
        K22 = self._noisify_kernel_matrix(K22_, self.noise)
        K = torch.vstack([torch.hstack([K11, K12]),
                          torch.hstack([K21, K22])])
        U11 = self.upper_chol_matrix
        U12, _ = torch.triangular_solve(K12, U11,
                                        upper=True,
                                        transpose=True)  # (m,1)
        U21 = torch.zeros(K21.shape)
        U22 = torch.linalg.cholesky(
            K22 - U12.transpose(-2, -1)@U12).transpose(-2, -1)
        U = torch.vstack([torch.hstack([U11, U12]),
                          torch.hstack([U21, U22])])
        self.ndata += nnew
        self.X = Xup
        self.y_ = yup - self.ymax
        self.kernel_matrix = K
        self.upper_chol_matrix = U

    def downdate(self, drop_inds):
        """Downdate data. Not implemented"""
        raise NotImplementedError

    def kernel_function(self, X1, X2, diagonal=False):
        """
        Calculate kernel function of GP

        Parameters
        ----------
        X1 : List[float]
            First kernel argument
        X2 : List[float]
            Second kernel argument
        diagonal : bool
            If False, return pairwise kernel evaluations tensor
            If True, return diagonal of pairwise kernel evaluations tensor

        Returns
        -------
        torch.Tensor
            Kernel values

        """
        return self._make_kernel_matrix(X1, X2,
                                        self.theta,
                                        self.lengthscale,
                                        diagonal=diagonal)

    def set_params_empirical_(self, X, y):
        """
        Set initialize params empirically from data (quick and dirty)

        Parameters
        ----------
        X : [float]
            Data matrix for GP
        y : [float]
            Evaluation vector for GP

        """
        mean = torch.mean(y) if 'mean' not in self.fixed_params else self.mean
        theta = torch.sqrt(torch.mean((y-mean)**2))  # Biased, but whatever
        horizontal_scale = (torch.max(X, dim=0).values -
                            torch.min(X, dim=0).values)
        lengthscale = horizontal_scale/3.0
        if 'mean' not in self.fixed_params:
            self.mean = mean
        if 'theta' not in self.fixed_params:
            self.theta = theta
        if 'lengthscale' not in self.fixed_params:
            self.lengthscale = lengthscale

    def kernel_matrix_condition_number(self, p=None):
        """Condition number of the kernel matrix"""
        return torch.linalg.cond(self.kernel_matrix, p=p)
    
    def fix_noise(self):
        """Fix noise for optimization"""
        self.fixed_params.add('noise')

    def fix_mean(self):
        """Fix mean for optimization"""
        self.fixed_params.add('mean')

    def unfix_noise(self):
        """Unfix noise for optimization"""
        self.fixed_params.discard('noise')

    def unfix_mean(self):
        """Unfix mean for optimization"""
        self.fixed_params.discard('mean')

    @property
    def mean(self):
        """torch.Tensor : Mean of GP"""
        return self._mean

    @property
    def theta(self):
        """torch.Tensor : Kernel output scale"""
        return self._derawfy(self._raw_theta)

    @property
    def lengthscale(self):
        """torch.Tensor : Kernel lengthscale"""
        return self._derawfy(self._raw_lengthscale)

    @property
    def noise(self):
        """torch.Tensor : Noise of GP"""
        return self._derawfy(self._raw_noise)

    @theta.setter
    def theta(self, x):
        x = utils.tensor_convert(x)
        self._raw_theta = self._rawfy(x)
        if self.kernel_matrix is not None: #Recalculate matrixes
            self.set_data(self.X, self.y)

    @mean.setter
    def mean(self, x):
        x = utils.tensor_convert(x)
        self._mean = x

    @lengthscale.setter
    def lengthscale(self, x):
        x = utils.tensor_convert(x)
        self._raw_lengthscale = self._rawfy(x)
        if self.kernel_matrix is not None: #Recalculate matrixes
            self.set_data(self.X, self.y)
            
    @noise.setter
    def noise(self, x):
        x = utils.tensor_convert(x)
        x = torch.clamp(x, 1e-20, None)  # In order to avoid -infs for rawfy
        self._raw_noise = self._rawfy(x)
        if self.kernel_matrix is not None: #Recalculate matrixes
            self.set_data(self.X, self.y)

    @property
    def y(self):
        """torch.Tensor : Evaluation tensor"""
        return self.y_ + self.ymax

    def _predict(self, xpred, to_return='mean'):
        # a^T K^-1 b = a^T (U^T U)^-1 b= (U^-T a)^T (U^-T b)
        if len(xpred.shape) == 1:
            xpred = torch.unsqueeze(xpred, 0)  # (n,d)
        kxpred = self._make_kernel_matrix(
            self.X, xpred, self.theta, self.lengthscale)  # (m,n)
        y_, _ = torch.triangular_solve(self.y_-self.mean,
                                       self.upper_chol_matrix,
                                       upper=True,
                                       transpose=True)  # (m,1)
        kxpred_, _ = torch.triangular_solve(kxpred,
                                            self.upper_chol_matrix,
                                            upper=True,
                                            transpose=True)  # (m,n)
        pred_mean = (kxpred_.transpose(-2, -1)@y_) + self.mean + self.ymax
        pred_mean = torch.squeeze(pred_mean, -1)
        if to_return == 'mean':
            return pred_mean
        elif to_return == 'cov':
            Kxpxp = self._make_kernel_matrix(
                xpred, xpred, self.theta, self.lengthscale)
            Kxpxp = self._noisify_kernel_matrix(Kxpxp, self.noise)
            pred_cov = Kxpxp - \
                kxpred_.transpose(-2, -1)@kxpred_
            return pred_mean, pred_cov
        elif to_return == 'var':
            Kxpxp = self._make_kernel_matrix(xpred, xpred, self.theta, self.lengthscale,
                                             diagonal=True)
            Kxpxp += self.noise**2 + self.min_jitter
            pred_var = Kxpxp - (kxpred_.transpose(-2, -1)**2).sum(dim=-1)
            return pred_mean, pred_var

    def _rawfy(self, x):
        return utils.invsoftplus(x)  # invsoftmax

    def _derawfy(self, y):
        return utils.softplus(y)  # softmax

    def _make_kernel_matrix(self, X1, X2, theta, lengthscale,
                            diagonal=False):
        output = 'pairwise' if not diagonal else 'diagonal'
        K = kernelfunctions.kernel_function(X1, X2,
                                            theta=theta,
                                            l=lengthscale,
                                            kind=self.kind,
                                            output=output)
        return K

    def _noisify_kernel_matrix(self, kernel_matrix, noise):
        K_ = utils.jittering(kernel_matrix, noise**2+self.min_jitter)
        return K_

    def _make_cholesky(self, K):
        U = torch.linalg.cholesky(K).transpose(-2, -1)  # Lower to upper
        return U

    def _loglikelihood_wrapper(self, params):
        theta = self._derawfy(params.get('raw_theta', self._raw_theta))
        noise = self._derawfy(params.get('raw_noise', self._raw_noise))
        mean = params.get('mean', self.mean)
        lengthscale = self._derawfy(params.get(
            'raw_lengthscale', self._raw_lengthscale))
        # Used for maximization in minimizer
        res = -self._loglikelihood(theta, lengthscale, noise, mean)
        return res

    def _loglikelihood(self, theta, lengthscale, noise, mean):
        kernel_matrix_ = self._make_kernel_matrix(
            self.X, self.X, theta, lengthscale)
        kernel_matrix = self._noisify_kernel_matrix(kernel_matrix_, noise)
        upper_chol_matrix = self._make_cholesky(kernel_matrix)
        y_, _ = torch.triangular_solve(self.y_-mean,
                                       upper_chol_matrix,
                                       upper=True,
                                       transpose=True)  # (m,1)
        term1 = -0.5*torch.sum(y_**2)
        term2 = -torch.sum(torch.log(torch.diagonal(upper_chol_matrix)))
        term3 = -0.5*self.ndata*math.log(2*math.pi)
        return term1 + term2 + term3

    def _set_lengthscale(self, lengthscale, ard):
        lengthscale = utils.tensor_convert(lengthscale)
        if lengthscale.ndim > 0:
            lengthscale = torch.squeeze(lengthscale)
            assert lengthscale.ndim == 1
            assert lengthscale.shape[0] == self.dim
        elif ard:
            lengthscale = torch.ones(self.dim)*lengthscale
        else:
            pass
        return lengthscale
