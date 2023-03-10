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


class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier ):
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size , diag_cov = False):
        if diag_cov:
          # Use only diagonal of the covariance matrix when the matrix of covariance has not inverse
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
    
    def update_covariance_inverse(self, elite_weights , diag_cov) -> None:
        cov = torch.cov(elite_weights.T) + self.noise
        if diag_cov :
            self.cov = torch.inverse(torch.diag(torch.diag(cov))) + self.noise
        else :
            u  =  torch.linalg.cholesky(cov)
            self.cov =  torch.cholesky_inverse(u)+ self.noise
        


# Create the PPO Agent
def create_CEM_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    if isinstance(env_agent.action_space , gym.spaces.Discrete):
        action_agent = DiscreteActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
    elif isinstance(env_agent.action_space, gym.spaces.Box):
        action_agent = ContinuousDeterministicActor(
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
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1, vmax=cfg.algorithm.max_epochs)

    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "black"]
    # 1)  Build the  logger
    seed = cfg.algorithm.seed
    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)
    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    pop_size = cfg.algorithm.pop_size

    fig, axs = plt.subplots(1,3,figsize=(12, 5))
    if cfg.CEMi :
        fig.suptitle("CEMi evolution")
    else :
        fig.suptitle("CEM evolution")
    # 7) Training loop
    for run in range(cfg.algorithm.nb_runs):
        seed+=1
        torch.manual_seed(seed)
        eval_env_agent = create_no_reset_env_agent(cfg)
        
        eval_agent = create_CEM_agent(cfg, eval_env_agent)
        centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
        matrix = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
        )
        best_score = -np.inf
        nb_steps = 0
        scores_elites = [] #scores des elites a chaque epoch (generation)
        for epoch in range(cfg.algorithm.max_epochs):
            print(cfg.algorithm.max_epochs*run + epoch)
            matrix.update_noise()
            scores = []
            weights = matrix.generate_weights(centroid, pop_size ,cfg.diag_cov)

            #generate pop_size generations
            for i in range(pop_size):
                workspace = Workspace()
                w = weights[i]
                torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())
                eval_agent(workspace, t=0, stop_variable="env/done")
                action = workspace["action"]
                nb_steps += action[0].shape[0]
                #stock the reward of each step in rewards
                rewards = workspace["env/cumulated_reward"][-1]
                #calculate the mean of all rewards 
                mean_reward = rewards.mean()
                logger.add_log("reward", mean_reward, nb_steps)
                scores.append(mean_reward)
            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
            elites_weights = [weights[k] for k in elites_idxs]
            #Concanetane the tensor of elites_weights
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            scores_elites.append([scores[i] for i in elites_idxs])
            centroid = elites_weights.mean(0)
            # Update covariance
            matrix.update_noise()
            if cfg.CEMi :
                matrix.update_covariance_inverse(elites_weights,cfg.diag_cov)
            else:
                matrix.update_covariance(elites_weights)
            scores_elites_np = np.asarray(scores_elites)
            for i in range(len(scores_elites_np)):
                Y  =  scores_elites_np[i,:]
                X = [i for j in range(len(scores_elites_np[0]))]
            axs[0].scatter(X,Y, color=colors[run] , s=3)

        X = [i for i in range(len(scores_elites))]
        meanY =  [np.mean(y) for y in scores_elites]
        medianY =  [np.median(y) for y in scores_elites]

        axs[1].plot(X,meanY,marker='o', color=colors[run] )
        axs[2].plot(X,medianY,marker='o', color=colors[run])
    
    for g in range(3):
        axs[g].set_xlabel('Generations')
        axs[g].grid(True)
    axs[0].set_ylabel('Score of elites')
    axs[1].set_ylabel('Mean Score of elites')
    axs[2].set_ylabel('Median Score of elites')
    axs[0].grid(False)
    plt.tight_layout()
    plt.show()


@hydra.main(
    config_path="./configs/",
    config_name="cem_CartPole.yaml",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg)


if __name__ == "__main__":
    main()
