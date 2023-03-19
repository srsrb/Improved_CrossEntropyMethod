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
            self.cov = inv_diagonal(torch.diag(torch.diag(cov))) + self.noise
        else :
            u  =  torch.linalg.cholesky(cov)
            self.cov =  torch.cholesky_inverse(u)+ self.noise
        

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

def rastrigin(x , y):
    A = 10
    return - (A*2 + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y)))
def ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi

    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return -(term1 + term2 + a + np.exp(1))

def goldstein_price(x, y):
    term1 = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    term2 = (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return -(term1 * term2)

def Three_hump_camel(x,y):
    return - (2*(x**2) - 1.05 * (x**4) +  (x**6)/6  + x*y + y**2)

def easom(x, y):
    return -(-np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2 - (y-np.pi)**2))

def run_cem(cfg, fonction_json):
    cmap = plt.cm.rainbow
    
    n_std = 1
    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)

    fig, ax = plt.subplots(1,2,figsize=(12, 5))
    pop_size = cfg.algorithm.pop_size

    for run in range(cfg.algorithm.nb_runs):
        
        maxX = 5
        minX = -5
        maxY = 5
        minY = -5
        centroid = torch.tensor(cfg.algorithm.centroid)
        #print("centroid : ", centroid)
        matrix = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
            cfg
        )
        if cfg.CEMi :
            fig.suptitle("iCEM evolution of "+cfg.algorithm.score_function+" function (centroid = " f"{cfg.algorithm.centroid})")
        else :
           fig.suptitle("CEM evolution of "+cfg.algorithm.score_function+" function (centroid = " f"{cfg.algorithm.centroid})")
        #ax = plt.gca()
        
        for epoch in range(cfg.algorithm.max_epochs):
            print(f'Simulating {run}.{epoch} \n')
            # The scores are initialized
            scores = np.zeros(pop_size)
            # The params of policies at iteration t+1 are drawn according to a multivariate 
            # Gaussian whose center is centroid and whose shaoe is defined by cov
            weights = matrix.generate_weights(centroid, pop_size)
            #print("weighths : ",weights)

            #generate pop_size generations
            for i in range(pop_size):
                if cfg.algorithm.score_function == "score_func":
                    scores[i] = score_func(weights[i])
                elif cfg.algorithm.score_function == "rastrigin":
                    scores[i] = rastrigin(weights[i][0] , weights[i][1])
                elif cfg.algorithm.score_function == "ackley":
                    scores[i] = ackley(weights[i][0] , weights[i][1])
                elif cfg.algorithm.score_function == "goldstein_price":
                    scores[i] = goldstein_price(weights[i][0] , weights[i][1])
                elif cfg.algorithm.score_function == "Three_hump_camel":
                    scores[i] = Three_hump_camel(weights[i][0] , weights[i][1])
                elif cfg.algorithm.score_function =="easom" :
                    scores[i] = easom(weights[i][0] , weights[i][1])
                else:
                    raise ValueError(cfg.algorithm.score_function," function is not implemented !!")

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
            """if (minX > np.min(w[:,0])):
                minX = np.min(w[:,0])
            if (minY > np.min(w[:,1])):
                minY = np.min(w[:,1])
            if (maxX < np.max(w[:,0])):
                maxX = np.max(w[:,0])
            if (maxY < np.max(w[:,1])):
                maxY = np.max(w[:,1])"""
            matrix.plot_gen(ax[0], w, c,n_std, epoch, facecolor='none', edgecolor=cmap(norm(epoch)))
        x = np.linspace(minX, maxX, 100)
        y = np.linspace(minY, maxY, 100)
        X, Y = np.meshgrid(x, y)
        if cfg.algorithm.score_function == "rastrigin":
            Z = rastrigin(X, Y)
        elif cfg.algorithm.score_function == "ackley":
            Z = ackley(X, Y)
        elif cfg.algorithm.score_function == "goldstein_price":
            Z = goldstein_price(X, Y)
        elif cfg.algorithm.score_function == "Three_hump_camel":
            Z = Three_hump_camel(X, Y)
        elif cfg.algorithm.score_function =="easom" :
            Z = easom(X, Y)
        else:
            raise ValueError(cfg.algorithm.score_function," function is not implemented !!")
        

        sns.heatmap(Z, cmap="rainbow")
    
        ax[1].set_xticks(np.linspace(0, 99, 5))
        ax[1].set_xticklabels(np.linspace(minX, maxX, 5))
        ax[1].set_yticks(np.linspace(0, 99, 5))
        ax[1].set_yticklabels(np.linspace(minY, maxY, 5))
        ax[1].set_xlim(0, 99)
        ax[1].set_ylim(0, 99)
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