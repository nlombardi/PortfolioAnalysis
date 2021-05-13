import pandas as pd
import torch
import numpy as np

# TODO: Solve derivatives/partial derivatives in Molina paper with RFR in the Sharpe Ratio


def sharpe(rets):
    return (rets.mean())/rets.std()


def positions(x, theta):
    # number of time series inputs
    M = len(theta) - 2
    # length of price series (time)
    T = len(x)
    # empty array to store positions --> Ft=tanh(theta*Txt) where xt=[1,rt-M,...,rt,Ft-1] and rt is chg in val t / t-1
    Ft = np.zeros(T)
    for t in range(M, T):
        xt = np.concatenate([1], x[t-M:t], [Ft[t-1]])
        Ft[t] = np.tanh(np.dot(theta, xt))

    return Ft


def returns(Ft, x, delta):
    T = len(x)
    # calculate returns at each time step using Rt = Ft-1*rt - delta*|Ft-Ft-1|
    rets = Ft[0:T-1]*x[1:T]-delta*np.abs(Ft[1:T]-Ft[0:T-1])

    return np.concatenate([[0], rets])


def gradient(x, theta, delta):
    # See study by Grabriel Molina: Stock Trading with Recurrent Reinforcement Learning (RRL) for derivatives/partials
    T = len(x)
    M = len(theta) - 2
    Ft = positions(x, theta)
    R = returns(Ft, x, delta)

    A = np.mean(R)
    B = np.mean(np.square(R))
    S = A/np.sqrt(B-A**2)

    dSdA = S*(1+S**2)/A
    dSdB = -S**3/2/A**2
    dAdR = 1/T
    dBdR = 2/T*R

    grad = np.zeros(M+2) # initialize the gradient
    dFpdtheta = np.zeros(M+2) # for storing previous dFdtheta

    for t in range(M, T):
        xt = np.concatenate([[1], x[t-M:t], [Ft[t-1]]])
        dRdF = -delta * np.sign(Ft[t]-Ft[t-1])
        dRdFp = x[t] + delta * np.sign(Ft[t]-Ft[t-1])
        dFdtheta = (1-Ft[t]**2) * (xt + theta[-1] * dFpdtheta)
        dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
        grad = grad + dSdtheta
        dFpdtheta = dFdtheta

    return grad, S


def train(x, epochs=2000, M=8, commission=0.0025, learning_rate=0.3):
    # will update out theta for each epoch using theta = theta + alpha(dStdtheta) alpha: learning rate, theta: weights
    theta = np.random.rand(M+2)
    sharpes = np.zeros(epochs) # store sharpe ratios over time
    for i in range(epochs):
        grad, sharpe = gradient(x, theta, commission)
        theta = theta + grad * learning_rate
        sharpes[i] = sharpe

    print("finished training")
    return theta, sharpes


def main():
    prices = pd.read_csv("./data/mvis.csv").set_index("Datetime")
