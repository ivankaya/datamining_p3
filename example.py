'''
Created on Dec 7, 2016

@author: ruggiero
'''
import numpy as np
import scipy as sp
import scipy.spatial as spatial
#import sklearn as sk


k=20

def mapper(key, value):
    global k
    # key: None
    # value: one line of input file

    alpha=16*np.log(k+2)
    B=mel_sampling(value,k)

    yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(200, 250)


def mel_sampling(X,k):
    B=np.zeros([k,250],dtype='float')
    B[0]=X[np.random.random_integers(0,len(X))]
    mel_prob=np.zeros([len(X)],dtype='float')
#   mel_dist=np.zeros([len(X),k],dtype='float')
    s=0
    a=range(len(X))
    
    covariance=np.zeros([len(B[0]),len(B[0])], dtype='float')
    np.fill_diagonal(covariance,1)
    for i in range(1,k):
        s=0
        for j,el in enumerate(X):
            dist=np.inf
            #s=0
            for l,mu in enumerate(B):
                if mu[0]==0 and mu[1]==0:
                    continue
                mel=spatial.distance.mahalanobis(el,mu,covariance)
                
                if mel<dist:
                    dist=mel
                    
            mel_prob[j]=dist
            s+=dist
            #assert sum(mel_prob)==s
        
        #Calculate probabilities
        print sum(mel_prob),s
        mel_prob=mel_prob/s
  

  
        #Add element to B

        r=np.random.choice(a,p=mel_prob)
        B[i]=X[r]
        
    return B,mel_prob

    

