import random
import numpy as np
import pandas as pd

def normalize(belief):   return belief/sum(belief)

class Belief():
    def __init__(self, bins, correct):
        self.correct = correct # set correct rate function

        self.bins, self.rank = bins, bins.shape[1] # TODO : bins to rank
        self.belief = normalize( np.ones(self.P.shape[0]) ) # initial belief, uniform 
        
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], [] # record Sigma at each step
        self.laplace_approx()

    def laplace_approx(self):
        self.theta_hat = 
        self.Sigma = 

        self.theta_hat_t[self.t], self.Sigma_t[self.t] = self.theta_hat, self.Sigma
        self.t += 1

    def initialize(self): # clear history
        self.belief = normalize( np.ones(self.P.shape[0]) )
        self.t, self.theta_hat_t, self.Sigma_t = 0, [], []
        self.laplace_approx()

    def imagine_update(self, x, y): # expect next update
        r_hat = self.correct(self.theta_hat, x)
        return normalize( self.belief * (r_hat if y == True else (1-r_hat)) ) # return next belief

    def update(self, x, y): # realization of update
        self.belief = self.imagine_update(self, x, y)
        self.laplace_approx()
