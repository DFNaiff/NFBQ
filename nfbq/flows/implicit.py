# -*- coding: utf-8 -*-

import torch


def newton_step(x, f, df, *args):
    return x - f(x, *args)/df(x, *args)


def newton_method(f, df, *args,
                  x0 = None, maxiter=100,
                  tol = 1e-6):
    if x0 is None:
        x = torch.randn_like(sum(args))
    else:
        x = x0
    for i in range(maxiter):
        xnew = newton_step(x, f, df, *args)
        if torch.max(torch.abs(x - xnew)) < tol:
            break
        else:
            x = xnew
    return x