# -*- coding: utf-8 -*-
import torch


class NormalizingFlow(torch.nn.Module):
    """
        Normalizing flow network, for modeling arbitrary
        probability distributions.
        
        Attributes
        ----------
        base_distrib : BaseDistrib
            The base distribution at the flow beginning
        flows : ModuleList[NormalizingLayer]
            List of flow transformations
        has_logprob : bool
            Whether we can compute the log-density at flow end
    """
    def __init__(self, base_distrib, flows):
        """
        Parameters
        ----------
        base_distrib : BaseDistrib
            The base distribution at the flow beginning
        flows: List[NormalizingLayer]
            List of flow transformations
        
        """
        super().__init__()
        self.base_distrib = base_distrib
        self.flows = torch.nn.ModuleList(flows)
        self.has_logprob = self._check_inverse()
        
    def sample(self, nsamples):
        """
        Sample from normalizing flow
        
        Parameters
        ----------
        nsamples : int
            Number of samples
            
        Returns
        -------
        torch.Tensor
            Samples of flow distribution
        """
        
        samples = self.base_distrib.sample(nsamples)
        samples = self.flow_forward(samples)
        return samples
    
    def flow_forward(self, x):
        """
        Flow forward the input
        
        Parameters
        ----------
        x : torch.Tensor
            Value at base of the flow.

        Returns
        -------
        torch.Tensor
            Transformed flow.

        """
        for flow in self.flows:
            x = flow.forward(x)
        return x
    
    def ulogprob(self, x):
        """
        Flow forward the input and calculate density at end
        
        Parameters
        ----------
        x : torch.Tensor
            Value at base of the flow.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Value at the end of the flow and associated log-density

        """
        y = self.base_distrib.logprob(x)
        for flow in self.flows:
            y -= flow.logabsdet(x)
            x = flow.forward(x)
        return x, y
    
    def ulogprob_autodiff(self, x):
        """
        Flow forward the input and calculate density at end,
        using automatic differentiation. Very inefficient,
        use only for sanity checks
        
        Parameters
        ----------
        x : torch.Tensor
            Value at base of the flow.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Value at the end of the flow and associated log-density

        """
        y = self.base_distrib.logprob(x)
        for flow in self.flows:
            y -= flow.logabsdet(x)
            x = flow.forward(x)
        return x, y
        jac = torch.autograd.functional.jacobian(self.flow_forward, x)
        transform_term = -torch.log(torch.abs(torch.det(jac)))
        base_term = self.base_distrib.logprob(x)
        y = base_term + transform_term
        x = self.flow_forward(x)
        return x, y
    
    def logprob(self, x):
        """
        Log-density of the flow distribution
        
        Parameters
        ----------
        x : torch.Tensor
            Input value (at the flown domain)

        Returns
        -------
        torch.Tensor
            Associated log-density

        """
        if not self.has_logprob:
            raise NotImplementedError("Some layer does not have an inverse function")
        y = torch.zeros_like(x[..., 0])
        for flow in reversed(self.flows):
            x = flow.inverse(x)
            y -= flow.logabsdet(x)
        y += self.base_distrib.logprob(x)
        return y

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
        torch.Tensor, torch.Tensor
            Samples of flow distribution and log-densities
        """
        x = self.base_distrib.sample(nsamples)
        x, y = self.ulogprob(x)
        return x, y
    
    def elbo(self, logdensity, nsamples):
        """
        Monte Carlo estimation of the evidence lower bound 
        between the log density and the flow distribution
        
        Parameters
        ----------
        logdensity : Callable[torch.Tensor, torch.Tensor]
            Density function
        nsamples : int
            Number of samples for MC estimator
        
        Returns
        -------
        torch.Tensor
            Estimated value of ELBO
        """
        #Estimate of elbo
        x, qdensity = self.sample_and_logprob(nsamples)
        pdensity = logdensity(x)
        return (pdensity - qdensity).mean()
    
    def _check_inverse(self):
        for f in self.flows:
            if not f.has_inverse:
                return False
        return True