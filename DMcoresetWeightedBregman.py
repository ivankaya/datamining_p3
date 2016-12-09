# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:57:49 2016

@author: Siggi
"""

import numpy as np
from sklearn.cluster import KMeans

k=200
m = 500

def mapper(key, value):
    global k
    global m
    # key: None
    # value: one line of input file
    X = value
    alpha=16.0*(np.log2(k)+2.0)
    B=d2sampling(X)
    n = np.shape(X)[0]
    classi = np.ones(n)
    val = 1.0*np.ones(n)
    for i in range(n):
        temp =  closestB(X[i,:],B)
        classi[i] = temp[0]
        val[i] = temp[1]
    ctheta = (1.0/n)*np.sum(val)
    classi = classi.astype('int')
    s = np.ones(n)
    subsums = np.ones(k)
    for i in range(k):
        #print(val[np.where(classi==i)])
        subsums[i] = np.sum(val[np.where(classi==i)])/np.sum((classi==i).astype('int'))
    for i in range(n):
        s[i] = (alpha*val[i]+2.0*alpha*subsums[classi[i]])/ctheta+4.0*n/(1.0*np.sum((classi==classi[i]).astype('int')))
    p = s/np.sum(s)
    C = np.zeros([m,250+1],dtype='float')
    for i in range(m):
        uni = np.random.uniform(0,1,1)[0]
        sim = 0
        cnt = 0
        while(sim<uni):
            sim = p[cnt]+sim
            cnt = cnt+1
        C[i,0] = (1.0/(m*p[cnt-1]))
        C[i,1:] = X[cnt-1,:]
    yield 1,C  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    global k
    means = d2sampling(values[:,1:])
    oldmu = np.zeros([k,250],dtype='float')
    newmu = means
    np.shape(newmu)
    eps = 0.00001
    n = np.shape(values)[0]
    classi = np.ones(n)
    pi = np.zeros(k)
    pix = np.zeros([k,250],dtype='float')
    while(np.square(oldmu - newmu).sum(axis=1).sum()>eps):
        oldmu = newmu
        for i in range(np.shape(values)[0]):
            classi[i] = np.argmin(np.square(values[i,1:] - newmu).sum(axis=1))
        classi = classi.astype('int')
        pi = np.zeros(k)
        pix = np.zeros([k,250],dtype='float')
        for i in range(n):
            pi[classi[i]] += values[classi[i],0]
            pix[classi[i],:] += values[classi[i],0]*values[classi[i],1:]
        for i in range(k):
            newmu[i,:] = (1.0/pi[i])*pix[i]
    yield newmu

def closestB(x,B):
    mini = float("inf")
    ind = [0,0]
    for i in range(np.shape(B)[0]):    
       temp = np.inner(x-B[i,:],x-B[i,:]) 
       if (temp<mini):
            mini = temp
            ind[0] = i
    ind[1] = mini
    return ind
def dA(x,b,dvec):
    for i in range(np.shape(x)[0]):
        temp = np.inner(x[i,:]-b,x[i,:]-b)
        if (temp<dvec[i]):
            dvec[i] = temp
    return dvec
    
def d2sampling(X):
    global k
    B = np.zeros([k,250],dtype='float')
    n = np.shape(X)[0]
    ind = np.random.randint(0,n,1)[0]
    B[0,:] = X[ind,:]
    probs = float("inf")*np.ones(n)
    for i in range(k-1):
        probs = dA(X,B[i,:],probs)
        probss = probs/(np.sum(probs))
        uni = np.random.uniform(0,1,1)[0]
        sim = 0
        cnt = 0
        while(sim<uni):
            sim = probss[cnt]+sim
            cnt = cnt+1
        B[i+1,:] = X[cnt-1,:]
        #print(cnt)
    return B


