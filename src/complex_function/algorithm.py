from turtle import update
import numpy as np


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
import sys



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

def CEM_combine(test_function , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb ) :
        """Returns:
        (all_weights,all_centroids,all_covs)
        """
        
        all_weights = []
        all_centroids= [centroid]
        percentage_CEM = []
        percentage_CEMir = []
        size = pop_size//2
        
        torch.manual_seed(seed) 
        centroid = torch.tensor(centroid)
        matrix_CEM = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        matrix_CEMir = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        all_covs = [matrix_CEM.get_cov().detach().cpu().numpy()] + [matrix_CEMir.get_cov().detach().cpu().numpy()]
        for epoch in range(max_epochs):
            print(f'Simulating {epoch} \n',end='\r')
            sys.stdout.write("\033[F")
            # The scores are initialized
            scores_CEM = np.zeros(size)
            scores_CEMir = np.zeros(size)
            # The params of policies at iteration t+1 are drawn according to a multivariate 
            # Gaussian whose center is centroid and whose shaoe is defined by cov
            weights_CEM = matrix_CEM.generate_weights(centroid, size)
            
            weights_CEMir = matrix_CEMir.generate_weights(centroid, size)
            weights = weights_CEM + weights_CEMir
            #generate pop_size generations
            for i in range(size):
                scores_CEM[i] = test_function(weights_CEM[i])
            for i in range(size ):
                scores_CEMir[i] = test_function(weights_CEMir[i])
            matrix_CEM.update_noise()
            matrix_CEMir.update_noise()
            # Keep only best individuals to compute the new centroid
            scores = np.concatenate((scores_CEM,scores_CEMir), axis=0)
            elites_idxs = np.argsort(scores)[:elites_nb]
            elites_weights = [weights[k] for k in elites_idxs]
            common_CEM = intersection(elites_weights , weights_CEM)
            common_CEMir = intersection(elites_weights , weights_CEMir)

            percentage_CEM.append((common_CEM * 100)/elites_nb)
            percentage_CEMir.append((common_CEMir * 100)/elites_nb)
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            centroid = elites_weights.mean(0)
            matrix_CEM.update_covariance_inverse_resize(elites_weights)
            matrix_CEMir.update_covariance(elites_weights)

            all_weights.append(np.array([t.detach().cpu().numpy() for t in weights]))
            all_centroids.append(centroid.detach().cpu().numpy())
            all_covs.append(matrix_CEM.get_cov().detach().cpu().numpy())
            all_covs.append(matrix_CEMir.get_cov().detach().cpu().numpy())
        all_weights, all_centroids,all_covs , percentage_CEM , percentage_CEMir = np.array(all_weights),np.array(all_centroids),np.array(all_covs) , np.array(percentage_CEM) , np.array (percentage_CEMir)

        return all_weights,all_centroids,all_covs, percentage_CEM , percentage_CEMir


def CEM_combine_2(test_function , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb ) :
        """Returns:
        (all_weights,all_centroids,all_covs)
        """
        
        all_weights = []
        all_centroids= [centroid]
        size = pop_size//3
        
        torch.manual_seed(seed) 
        centroid = torch.tensor(centroid)
        matrix_CEM = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        matrix_CEMir = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        matrix_CEMi = CovMatrix(
            centroid,
            sigma,
            noise_multiplier,
        )
        all_covs = [matrix_CEM.get_cov().detach().cpu().numpy()] + [matrix_CEMir.get_cov().detach().cpu().numpy()] + [matrix_CEMi.get_cov().detach().cpu().numpy()]
        for epoch in range(max_epochs):
            print(f'Simulating {epoch} \n',end='\r')
            sys.stdout.write("\033[F")
            # The scores are initialized
            scores = np.zeros(3*size)
            # The params of policies at iteration t+1 are drawn according to a multivariate 
            # Gaussian whose center is centroid and whose shaoe is defined by cov
            weights_CEM = matrix_CEM.generate_weights(centroid, size)
            
            weights_CEMir = matrix_CEMir.generate_weights(centroid, size)
            weights_CEMi = matrix_CEMi.generate_weights(centroid, size)
            
            weights = weights_CEM + weights_CEMir + weights_CEMi
            #generate pop_size generations
            for i in range(3*size):
                scores[i] = test_function(weights[i])
            matrix_CEM.update_noise()
            matrix_CEMir.update_noise()
            matrix_CEMi.update_noise()
            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[:elites_nb]
            elites_weights = [weights[k] for k in elites_idxs]
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            centroid = elites_weights.mean(0)
            matrix_CEM.update_covariance_inverse_resize(elites_weights)
            matrix_CEMir.update_covariance(elites_weights)
            matrix_CEMi.update_covariance(elites_weights)

            all_weights.append(np.array([t.detach().cpu().numpy() for t in weights]))
            all_centroids.append(centroid.detach().cpu().numpy())
            all_covs.append(matrix_CEM.get_cov().detach().cpu().numpy())
            all_covs.append(matrix_CEMir.get_cov().detach().cpu().numpy())
            all_covs.append(matrix_CEMi.get_cov().detach().cpu().numpy())
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

def intersection (A , B) :
    count = 0
    for a in A:
        for b in B:
            if torch.equal(a, b):
                count += 1
    return count 