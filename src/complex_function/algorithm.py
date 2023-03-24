from turtle import update
import numpy as np


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
import sys

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

    def get_cov(self):
        return torch.clone(self.cov)
    
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

    def update_covariance_inverse_resize(self,elite_weights)->None: 
        cov = torch.cov(elite_weights.T) + self.noise
        u  =  torch.linalg.cholesky(cov)
        cov_i =   torch.cholesky_inverse(u)
        
        eig_cov_max = torch.linalg.eigh(cov)[0][-1]
        eig_cov_i_max = torch.linalg.eigh(cov_i)[0][-1]
        self.cov  =  (cov_i*eig_cov_max/eig_cov_i_max)+ self.noise

    
def CEM(test_function , EXPERIMENT, CEMi , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb ) :
        """Returns:
        (all_weights,all_centroids,all_covs)
        """
        
        all_weights = []
        all_centroids= [centroid]
        
        
        torch.manual_seed(seed) 
        centroid = torch.tensor(centroid)
        matrix = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        all_covs = [matrix.get_cov().detach().cpu().numpy()]

        for epoch in range(max_epochs):
            print(f'Simulating {epoch} \n',end='\r')
            sys.stdout.write("\033[F")
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
            if EXPERIMENT:
                matrix.update_covariance_inverse_resize(elites_weights)
            elif CEMi :
                matrix.update_covariance_inverse(elites_weights)
            else:
                matrix.update_covariance(elites_weights)

            all_weights.append(np.array([t.detach().cpu().numpy() for t in weights]))
            all_centroids.append(centroid.detach().cpu().numpy())
            all_covs.append(matrix.get_cov().detach().cpu().numpy())
        all_weights, all_centroids,all_covs = np.array(all_weights),np.array(all_centroids),np.array(all_covs)

        return all_weights,all_centroids,all_covs


def plot_confidence_ellipse(cov, ax,weights, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    cov : array_like, shape (n, )
        Input matrix.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    weights
        The weights 

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                       width=ell_radius_x * 2,
                        height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    x = weights[:,0]
    y = weights[:,1]

    # Compute the stdandard deviation of x from
    # the square root of the variance and multiply
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # compute the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
    .rotate_deg(45) \
    .scale(scale_x, scale_y) \
    .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

    