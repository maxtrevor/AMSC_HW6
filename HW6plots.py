import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import lambertw

p1dat = pickle.load( open( "p1dat.p", "rb" ) )
p2dat = pickle.load( open( "p2dat.p", "rb" ) )

zs = np.linspace(0.1,4,40) 
av_fracS = p1dat[0]
av_small = p1dat[1]

plt.figure()
plt.plot(zs,av_fracS,label='experimental')
th_S = 1+lambertw(-zs*np.exp(-zs))/zs
th_S[9] = 0
th_S = np.real(th_S)
plt.plot(zs,th_S,label='theoretical')
plt.xlabel('z')
plt.ylabel('S(z)')
plt.legend()

plt.figure()
th_s = 1/(1-zs+zs*th_S)
plt.plot(zs,av_small,label='experimental')
plt.plot(zs,th_s,label='theoretical')
plt.xlabel('z')
plt.ylabel('<s>')
plt.legend()


z = 4
qs = np.array([10,11,12,13])
ns = 2**qs

th_l = np.log(ns)/np.log(z)

plt.figure()
plt.plot(ns,p2dat,label='experimental')
plt.plot(ns,th_l,label='theoretical')
plt.xscale('log')
plt.xlabel('n')
plt.ylabel('average path length')
plt.legend()