from turtle import update
import numpy as np
import gym
import os


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.loggers import Logger
from bbrl_examples.models.actors import DiscreteActor
from bbrl_examples.models.envs import create_no_reset_env_agent

from bbrl.visu.visu_policies import plot_policy



class CovMatrix():
    """
    The covariance matrix of the Cross-Entropy Method
    """
    def __init__(self, sigma, noise_multiplier, seed, diag_cov=False):

      self.sigma = sigma
      self.noise_multiplier = noise_multiplier

      self.diag_cov = diag_cov
           
      self.rng = np.random.default_rng(seed)

    def init_covariance(self, centroid: np.ndarray) -> None:
      """
      Initialize the covariance matrix from a vector of parameters.
      If the centroid is (0,0), then it is very important that the noise matrix is not null. 
      Otherwise the covariance matrix is null and all generations consist of a single point.
      :param centroid: the vector of parameters where search starts
      """
      self.policy_dim = len(centroid)
      # self.noise_matrix = np.ones((self.policy_dim, self.policy_dim)) * self.sigma
      self.noise_matrix = np.diag(np.ones(self.policy_dim) * self.sigma)
      self.cov = np.diag(np.ones(self.policy_dim) * np.var(centroid)) + self.noise_matrix

    def update_noise(self) -> None:
      """
      Update the noise matrix with the noise multiplier
      """
      self.noise_matrix = self.noise_matrix * self.noise_multiplier

    def generate_weights(self, centroid, pop_size) -> np.ndarray:
      """
      Generate a set of individuals around the centroid from the current covariance matrix
      :param centroid: the vector of parameters obtained at the previous generation
      :param pop_size: the number of individuals of the next generation
      """
      if self.diag_cov:
          # Separable CEM, useful when self.policy_dim >> 100
          # Use only diagonal of the covariance matrix
          param_noise = np.random.randn(pop_size, self.policy_dim)
          print(np.diag(self.cov))
          weights = centroid + param_noise * np.sqrt(np.diagonal(self.cov)) ## pq std dev??
      else:
          # The params of policies at iteration t+1 are drawn according to a multivariate
          # Gaussian whose center is centroid and whose shape is defined by cov
          weights = self.rng.multivariate_normal(centroid, self.cov, pop_size)
      return weights

    def update_covariance(self, elites_weights: np.ndarray) -> None:
      """
      Update the covariance matrix from the elite individual of the current generation
      """
      self.cov = np.cov(elites_weights, rowvar=False) + self.noise_matrix

    def update_covariance_no_noise(self, elites_weights: np.ndarray) -> None:
      """
      Update the covariance matrix from the elite individual of the current generation
      without adding the noise matrix (useful to compare the behavior with/without noise)
      """
      self.cov = np.cov(elites_weights, rowvar=False)


    def update_covariance_inverse(self, elites_weights: np.ndarray) -> None:
      """
      Update the covariance matrix from the elite individual of the current generation
      But inverting the matrix to favor sampling in the direction of higher variation
      Notes: 
      - the matrix being symmetric, one can do better than calling standard inversion
      - in the case where we use a diagonal matrix, we can do even better
      """
      self.cov  =  np.linalg.inv(np.diag(np.diag(np.cov(elites_weights,rowvar=False)) + self.noise_matrix
      



# Create the PPO Agent
def create_CEM_agent(cfg, env_agent):

    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    action_agent = DiscreteActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    ev_agent = Agents(env_agent, action_agent)
    eval_agent = TemporalAgent(ev_agent)
    eval_agent.seed(cfg.algorithm.seed)

    return eval_agent


def make_gym_env(env_name):
    env = gym.make(env_name)
    return env


def run_cem(cfg):
    # 1)  Build the  logger
    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)
    
    eval_env_agent = create_no_reset_env_agent(cfg)

    pop_size = cfg.algorithm.pop_size

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0
    #on a créer enAgent et action agent
    eval_agent = create_CEM_agent(cfg, eval_env_agent)


    #init centroid
    centroid :  np.ndarray = torch.nn.utils.parameters_to_vector(eval_agent.parameters()).detach().numpy()


    matrix = CovMatrix(
        cfg.algorithm.sigma,
        cfg.algorithm.noise_multiplier,
        cfg.algorithm.seed,
        diag_cov= True
        
        
        
    )
    matrix.init_covariance(centroid)
    
    best_score = -np.inf
    nb_steps = 0

    #creation d'un fichier pour stocker les résultats 
    f = open("Pendulum_CEM_100.txt", "w")

    scores_elites = [] #scores des elites a chaque epoch (generation)
    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        print("generation : ", epoch)

        matrix.update_noise()
        scores = []
        weights = matrix.generate_weights(centroid, pop_size)
        
        for i in range(pop_size):
            
            workspace = Workspace()
            w = weights[i]
            w_copy  = torch.tensor(w,dtype=torch.float32) #necessaire si nn erreur de dtype
            torch.nn.utils.vector_to_parameters(w_copy, eval_agent.parameters())
            eval_agent(workspace, t=0, stop_variable="env/done") 
            action = workspace["action"]
            nb_steps += action[0].shape[0]
            rewards = workspace["env/cumulated_reward"][-1]
            mean_reward = rewards.mean()
            logger.add_log("reward", mean_reward, nb_steps)

            # ---------------------------------------------------
            scores.append(mean_reward)

          
        # Keep only best individuals to compute the new centroid
        elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
        elites_weights = np.array( [weights[k] for k in elites_idxs])
        
        scores_elites.append([scores[i] for i in elites_idxs])
        #print("elites = ", elites_weights)
        #print("\n")
        centroid = np.mean(elites_weights,0)
        #print("centroid = ", centroid)
        #print("\n")
        # Update covariance
        matrix.update_noise()
        #matrix.update_covariance(elites_weights)
        matrix.update_covariance_inverse(elites_weights)
    f.close()

    scores_elites = np.asarray(scores_elites)

    #----------------------- Plotting
    print(type(scores_elites))
    for i in range(len(scores_elites[0])):
        X = [i for i in range(len(scores_elites))]
        Y  =  scores_elites[:,i]
        plt.scatter(X,Y)
    
    plt.title('Evolution of the Elite Scores for Inverse - CEM')
    plt.xlabel('Generation')
    plt.ylabel('Score of elites')
    
    plt.text(0, 400, f'#elites_elected= {cfg.algorithm.elites_nb} ' + r'$\Sigma = $' + str(cfg.algorithm.sigma))
    plt.grid(True)
    plt.show()
    



@hydra.main(
    config_path="./configs/",
    config_name="cem_cartpole.yaml",
    #version_base="1.1",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg)


if __name__ == "__main__":
    main()