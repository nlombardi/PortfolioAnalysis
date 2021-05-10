import statsmodels
import torch


def sharpe(rets):
    return rets.mean()/rets.std()

