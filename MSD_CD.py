#!/usr/bin/env python
import sys
import numpy as np 
from simpletraj.dcd.dcd import DCDReader
from pytool import hoomd_xml
from sys import argv

def autocorrFFT(X):
	M=X.shape[0]
	Fx = np.fft.rfft(X.T[0], 2*M)
	Fy = np.fft.rfft(X.T[1], 2*M)
	Fz = np.fft.rfft(X.T[2], 2*M)
	corr = abs(Fx)**2 + abs(Fy)**2 + abs(Fz)**2
	res = np.fft.irfft(corr)
	res= (res[:M]).real
	return res

def msd_fft(X):
	M = X.shape[0]
	D = np.square(X).sum(axis=1)
	D = np.append(D, 0)
	S2 = autocorrFFT(X)
	S2 = S2 / np.arange(M, 0, -1)
	Q = 2 * D.sum()
	S1 = np.zeros(M)
	for m in range(M):
		Q = Q - D[m-1] -D[M-m]
		S1[m] = Q / (M-m)
	return S1 - 2 * S2


def msd_Correlation(allX):
    M = allX.shape[0]
    allFX = np.fft.rfft(allX, axis=0,n=M*2)
    corr = np.sum(abs(allFX)**2, axis=(1,-1))
    return np.fft.irfft(corr, n=2 * M)[:M].real/np .arange(M, 0, -1)


def msd_square(allX):
    M = allX.shape[0]
    S = np.square(allX).sum(axis=(1, -1))
    S = np.append(S, 0)  #S[-1] == S[M] == 0
    SS = 2 * S.sum()
    SQ = np.zeros(M)
    for m in range(M): 
        SS = SS - S[m-1] - S[M-m]
        SQ[m] = SS / (M - m)
    return SQ

xml = hoomd_xml(argv[1])
types = xml.nodes['type']
indexDPPC = (types == 'H') | (types == 'M') | (types == 'T')

dcd = DCDReader('particles_os.dcd')
allX = np.asarray([_[indexDPPC] for _ in dcd])

msd = (msd_square(allX) - 2 * msd_Correlation(allX)) / allX.shape[1]

timescale = 200
time = np.arange(msd.shape[0]) * timescale 
np.savetxt('msd.txt', np.c_[time, msd], fmt='%.8f')
# np.savetxt('msd.txt', np.vstack([np.arange(msd.shape[0]), msd]).T, fmt='%.6f')
