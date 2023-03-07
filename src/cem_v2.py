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
        


# Create the PPO Agent
def create_CEM_agent(cfg, env_agent ,seed):
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

    eval_agent.seed(seed)

    return eval_agent


def make_gym_env(env_name):
    env = gym.make(env_name)
    return env
def run_2(cfg, run ,eval_agent ,centroid_2 , axs , colors, first_round=False) :
        pop_size = cfg.algorithm.pop_size
        make_plot = cfg.algorithm.make_plot
        centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
        matrix_2 = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
            cfg
        )
        weights_2 = matrix_2.generate_weights(centroid, pop_size )
        c = centroid[2:]
        if first_round :
            centroid = centroid[:2]
            print("les 2 premiers param sont : ", centroid)
        else :
            centroid = torch.cat((centroid_2, c), dim=0)
        matrix = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
            cfg
        )
        best_score = -np.inf
        nb_steps = 0
        scores_elites = [] #scores des elites a chaque epoch (generation)
        for epoch in range(cfg.algorithm.max_epochs):
            print(f'Simulating {run}.{epoch} \n')
            matrix.update_noise()
            scores = []
            weights = matrix.generate_weights(centroid , pop_size )
            #generate pop_size generations
            for i in range(pop_size):
                workspace = Workspace()
                if first_round :
                    w = torch.cat((weights[i] , weights_2[i][2:]), dim=0)
                else:    
                    w = weights[i]
                #on va concatener à W le reste des param (W_2)
                torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())
                eval_agent(workspace, t=0, stop_variable="env/done")
                action = workspace["action"]
                nb_steps += action[0].shape[0]
                #stock the reward of each step in rewards
                rewards = workspace["env/cumulated_reward"][-1]
                #calculate the mean of all rewards 
                mean_reward = rewards.mean()
                scores.append(mean_reward)
            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
            if first_round :
                elites_weights = [weights[k][:2] for k in elites_idxs]
            else :
                elites_weights = [weights[k] for k in elites_idxs]
            #Concanetane the tensor of elites_weights
            elites_weights = torch.cat(
                [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            scores_elites.append([scores[i] for i in elites_idxs])
            m = elites_weights.mean(0)
            if first_round :
                centroid = m[:2]
            else :
                centroid = torch.cat((centroid_2, m[2:]), dim=0)

            # Update covariance
            matrix.update_noise()
            #je dois prendre que les 2 premier param de chaque element de la liste eights dans elite weights pour faire les update
            if cfg.CEMi and first_round :
                matrix.update_covariance_inverse(elites_weights)
            else:
                matrix.update_covariance(elites_weights)

            scores_elites_np = np.asarray(scores_elites)
            if not first_round :
                for i in range(len(scores_elites_np)):
                    Y  =  scores_elites_np[i,:]
                    X = [i for j in range(len(scores_elites_np[0]))]



                if make_plot : 
                    axs[0].scatter(X,Y, color=colors[run] , s=3)  
                
        if first_round :
            print("les 2 premiers param optimisé sont : ", centroid)
            return centroid 
        else :
            return scores_elites

def run_cem(cfg, fonction_json):
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1, vmax=cfg.algorithm.max_epochs)

    means =[0] * cfg.algorithm.nb_runs
    make_plot = cfg.algorithm.make_plot

    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "black"]
    # 1)  Build the  logger
    seed = cfg.algorithm.seed
    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)
    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    pop_size = cfg.algorithm.pop_size

    fig, axs = plt.subplots(1,3,figsize=(12, 5))
    title  = ''
    if cfg.CEMi :
        title  = "CEMi evolution" 
    else :
        title  = "CEM evolution"
    title  += (" - diagonal only" if cfg.algorithm.diag_covMatrix else "  - full matrix") +" :: " f"{cfg.algorithm.architecture.actor_hidden_size}"
    fig.suptitle(title)

    json  = make_json.JSON_Summary(['CartPole'],"CEM","")

    for run in range(cfg.algorithm.nb_runs):
        seed+=1
        torch.manual_seed(seed)
        eval_env_agent = create_no_reset_env_agent(cfg)
        
        eval_agent = create_CEM_agent(cfg, eval_env_agent , seed)
        cendroid_2 = run_2(cfg,run,eval_agent , None , axs , colors , True )
        scores_elites = run_2(cfg,run,eval_agent , cendroid_2 , axs , colors , False )
        X = [i for i in range(len(scores_elites))]
        meanY =  [np.mean(y) for y in scores_elites]
        medianY =  [np.median(y) for y in scores_elites]
        if make_plot:
            
            axs[1].plot(X,meanY,marker='o', color=colors[run] )
            axs[2].plot(X,medianY,marker='o', color=colors[run])

        means[run] = np.float64(meanY[-1])

    json.add_scores_to_environment('CartPole',means)
    json.end_sampling()

    if make_plot:
        for g in range(3):
            axs[g].set_xlabel('Generations')
            axs[g].grid(True)
        axs[0].set_ylabel('Score of elites')
        axs[1].set_ylabel('Mean Score of elites')   
        axs[2].set_ylabel('Median Score of elites')
        axs[0].grid(False)
        plt.tight_layout()
        name_file = "plot_output/plot_"+f"{cfg.gym_env.env_name}"+ ("_CEMI" if cfg.CEMi else "_CEM") + f"_R{cfg.algorithm.nb_runs}G{cfg.algorithm.max_epochs}Diag{cfg.algorithm.diag_covMatrix }.png"
        os.makedirs('./plot_output/')
        plt.savefig(name_file)


@hydra.main(
    config_path="./configs/",
    config_name="cem_Pendulum.yaml",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg, lambda *args: None)

if __name__ == "__main__":
    main()