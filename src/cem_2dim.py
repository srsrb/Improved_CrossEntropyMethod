from turtle import update
import numpy as np
import gym
import os


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt
import matplotlib.colors

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.loggers import Logger
from bbrl_examples.models.actors import DiscreteActor , ContinuousDeterministicActor
from bbrl_examples.models.envs import create_no_reset_env_agent

from bbrl.visu.visu_policies import plot_policy

import make_json

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.pyplot import figure

# Init of the figure where we plot
figure(figsize=(10, 10), dpi=80)

# Init of the colormap
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=1, vmax=40)
class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier ,cfg ):
        policy_dim = centroid.size()[0]
        self.policy_dim  = policy_dim
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier
        self.cfg  =  cfg

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size ):
        if self.cfg.algorithm.diag_covMatrix:
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
        if self.cfg.algorithm.diag_covMatrix :
            eigenvalues = torch.linalg.eigvals(cov)
            self.cov = torch.inverse(torch.diag(torch.diag(cov))) + self.noise
            eigenvaluesI = torch.linalg.eigvals(self.cov)
            self.cov *= (torch.max(eigenvalues) / torch.max(eigenvaluesI))
        else :
            
            eigenvalues = torch.linalg.eigvals(cov)
            print("eigenvalues = ",eigenvalues.real)
            u  =  torch.linalg.cholesky(cov)
            self.cov =  torch.cholesky_inverse(u)+ self.noise
            eigenvaluesI = torch.linalg.eigvals(self.cov)
            print("eigenvaluesI = ",eigenvaluesI.real)
            self.cov *= (torch.max(eigenvalues.real) / torch.max(eigenvaluesI.real))
        

    def plot_gen(self, ax, weights, centroid, n_std, index, **kwargs):
      """
      Plot the set of individuals of a generation and the corresponding ellipsoid
      Can only be used in the 2D case
      """
      assert self.cov.shape == (2, 2), "Wrong covariance size for plotting"
      
      pearson = self.cov[0, 1]/np.sqrt(self.cov[0, 0] * self.cov[1, 1])
      # Using a special case to obtain the eigenvalues of this
      # two-dimensional dataset.
      ell_radius_x = np.sqrt(1 + pearson)
      ell_radius_y = np.sqrt(1 - pearson)
      ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    **kwargs)
  
      x = weights[:,0]
      y = weights[:,1]
      ax.scatter(x, y, color=cmap(norm(index)), s=3)
      # Compute the stdandard deviation of x from
      # the square root of the variance and multiply
      # with the given number of standard deviations.
      scale_x = np.sqrt(self.cov[0, 0]) * n_std
      mean_x = np.mean(x)

      # compute the stdandard deviation of y ...
      scale_y = np.sqrt(self.cov[1, 1]) * n_std
      mean_y = np.mean(y)

      transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

      ellipse.set_transform(transf + ax.transData)
      ax.add_patch(ellipse)

      plt.scatter(centroid[0], centroid[1], c='red', s=3)


def score_func(vec):
  x_target = 20
  y_target = 30
  return 10-np.sqrt((x_target-vec[0])*(x_target-vec[0])+(y_target-vec[1])*(y_target-vec[1]))

def run_cem(cfg, fonction_json):
    cmap = plt.cm.rainbow
    
    n_std = 1
    #means =[0] * cfg.algorithm.nb_runs
    #make_plot = cfg.algorithm.make_plot

    #colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "black"]
    # 1)  Build the  logger

    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)
    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    pop_size = cfg.algorithm.pop_size

    for run in range(cfg.algorithm.nb_runs):
        
        
        centroid = torch.zeros(cfg.algorithm.policy_dim)
        #print("centroid : ", centroid)
        matrix = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
            cfg
        )
        if cfg.CEMi :
            plt.title("iCEM evolution")
        else :
            plt.title("CEM evolution")
        ax = plt.gca()
        
        for epoch in range(cfg.algorithm.max_epochs):
            #print(f'Simulating {run}.{epoch} \n')
            # The scores are initialized
            scores = np.zeros(pop_size)
            # The params of policies at iteration t+1 are drawn according to a multivariate 
            # Gaussian whose center is centroid and whose shaoe is defined by cov
            weights = matrix.generate_weights(centroid, pop_size)
            #print("weighths : ",weights)

            #generate pop_size generations
            for i in range(pop_size):
                scores[i] = score_func(weights[i])

            #print(scores)
            matrix.update_noise()
            
            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
            elites_weights = [weights[k] for k in elites_idxs]
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            #print("elites_weights : ",elites_weights)
            centroid = elites_weights.mean(0)
            #print("La nouvelle centroid : ",centroid)
            if cfg.CEMi :
                matrix.update_covariance_inverse(elites_weights)
            else:
                matrix.update_covariance(elites_weights)
            w = np.array([t.detach().cpu().numpy() for t in weights])
            c = centroid.detach().cpu().numpy()
            matrix.plot_gen(ax, w, c,n_std, epoch, facecolor='none', edgecolor=cmap(norm(epoch)))
        plt.show()



@hydra.main(
    config_path="./configs/",
    config_name="cem_2dim.yaml",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg, lambda *args: None)

if __name__ == "__main__":
    main()