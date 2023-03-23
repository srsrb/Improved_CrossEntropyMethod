from turtle import update
import numpy as np


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors


from bbrl_examples.models.loggers import Logger

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.pyplot import figure


class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier , diag_covMatrix = False  ):
        policy_dim = centroid.size()[0]
        self.policy_dim  = policy_dim
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier
        self.diag_covMatrix = diag_covMatrix

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size ):
        if self.diag_covMatrix :
          # Only use the diagonal of the covariance matrix when the matrix of covariance has not inverse
          policy_dim = centroid.size()[0]
          param_noise = torch.randn(pop_size, policy_dim)
          weights = centroid + param_noise * torch.sqrt(torch.diagonal(self.cov))
        else:
            # The params of policies at iteration t+1 are drawn according to a multivariate
            # Gaussian whose center is centroid and whose shape is defined by cov
            dist = torch.distributions.MultivariateNormal(
            centroid, covariance_matrix=self.cov
            )
            weights = [dist.sample() for _ in range(pop_size)]
        return weights

    def update_covariance(self, elite_weights) -> None:

        self.cov = torch.cov(elite_weights.T) + self.noise
    
    def update_covariance_inverse(self, elite_weights ) -> None:
        def inv_diagonal(mat: torch.Tensor) -> torch.Tensor:
            res =  torch.zeros(self.policy_dim,self.policy_dim)
            for i in range(self.policy_dim):
                if self.cov[i,i] ==0 :
                    raise Exception("Tried to invert 0 in the diagonal of the cov matrix")
                res[i][i] = 1/self.cov[i][i]
            return res

        cov = torch.cov(elite_weights.T) + self.noise
        if self.diag_covMatrix  :
            self.cov = inv_diagonal(torch.diag(torch.diag(cov))) + self.noise
        else :
            u  =  torch.linalg.cholesky(cov)
            self.cov =  torch.cholesky_inverse(u)+ self.noise
        

    
def CEM(test_function , CEMi , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb ):
        torch.manual_seed(seed) 
        centroid = torch.tensor(centroid)
        matrix = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        all_weights = []
        for epoch in range(max_epochs):
            print(f'Simulating {epoch} \n')
            # The scores are initialized
            scores = np.zeros(pop_size)
            # The params of policies at iteration t+1 are drawn according to a multivariate 
            # Gaussian whose center is centroid and whose shaoe is defined by cov
            weights = matrix.generate_weights(centroid, pop_size)
            #print("weighths : ",weights)

            #generate pop_size generations
            for i in range(pop_size):
                scores[i] = test_function(weights[i])
            matrix.update_noise()
            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[:elites_nb]
            elites_weights = [weights[k] for k in elites_idxs]
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            centroid = elites_weights.mean(0)
            if CEMi :
                matrix.update_covariance_inverse(elites_weights)
            else:
                matrix.update_covariance(elites_weights)
            all_weights.append(np.array([t.detach().cpu().numpy() for t in weights]))
        return all_weights


