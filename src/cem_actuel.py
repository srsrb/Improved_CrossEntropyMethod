
from turtle import update
import numpy as np
import gym
import os



import torch
import torch.nn as nn
#torch.set_default_tensor_type(torch.DoubleTensor)
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
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier ,cfg ):
        """
        

        cfg -> fichier de configuration utilise
        """
        policy_dim = centroid.size()[0]
        self.policy_dim  = policy_dim
        self.noise = torch.diag(torch.ones(policy_dim ) * sigma)
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

    def update_covariance_inverse_resize(self,elite_weights)->None: 
        cov = torch.cov(elite_weights.T) + self.noise 
        print("les valeurs propres sont : ",torch.linalg.eigvalsh(cov))
        if torch.all(torch.linalg.eigvalsh(cov) > 0):
            print("La matrice est définie positive")
        else:
            print("La matrice n'est pas définie positive.")
            
        
        u  =  torch.linalg.cholesky(cov)
        cov_i =   torch.cholesky_inverse(u)
        
        eig_cov_max = torch.linalg.eigh(cov)[0][-1]
        eig_cov_i_max = torch.linalg.eigh(cov_i)[0][-1]
        print("eig_cov_max ",eig_cov_max)
        print("eig_cov_i_max ",eig_cov_i_max)
        print("facteur ",eig_cov_max/eig_cov_i_max)
        self.cov  =  (cov_i * (eig_cov_max/eig_cov_i_max))+ self.noise
        
    

# Create the PPO Agent
def create_CEM_agent(cfg, env_agent , seed):
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


def run_cem(cfg, fonction_json):

    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1, vmax=cfg.algorithm.max_epochs)

    make_plot = cfg.make_plot

    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "black"]
    # 1)  Build the  logger
    seed = cfg.algorithm.seed
    torch.manual_seed(cfg.algorithm.seed)
    logger = Logger(cfg)
    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    pop_size = cfg.algorithm.pop_size

    fig, axs = plt.subplots(1,3,figsize=(12, 5))
    title  = ''
    if cfg.EXPERIMENT:
        title  = "CEMir evolution" 
    elif cfg.CEMi :
        title  = "CEMi evolution" 
    elif cfg.merge :
        title  = "CEMm evolution" 
    else :
        title  = "CEM evolution"
    title  += (" - diagonal only" if cfg.algorithm.diag_covMatrix else "  - full matrix") +" :: " f"{cfg.algorithm.architecture.actor_hidden_size}"
    fig.suptitle(title)
    
    max_run_and_epoch  =  str(cfg.algorithm.nb_runs-1)+'.'+str(cfg.algorithm.max_epochs-1) 

    for run in range(cfg.algorithm.nb_runs):
        seed+=1
        torch.manual_seed(seed)
        eval_env_agent = create_no_reset_env_agent(cfg)
        
        eval_agent = create_CEM_agent(cfg, eval_env_agent , seed)
        centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
        matrix = CovMatrix(
            centroid,
            cfg.algorithm.sigma,
            cfg.algorithm.noise_multiplier,
            cfg
        )
        if cfg.merge :
            pop_size = pop_size//2
            matrix_CEM = CovMatrix(
                centroid,
                cfg.algorithm.sigma,
                cfg.algorithm.noise_multiplier,
                cfg
            )


        nb_steps = 0
        scores_elites = [] #scores des elites a chaque epoch (generation)
        
        for epoch in range(cfg.algorithm.max_epochs):
            
            print(f'Simulating {run}.{epoch} /{max_run_and_epoch} ')
            matrix.update_noise()
            scores = []
            weights = matrix.generate_weights(centroid, pop_size )

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
            #print("scores CEMir ", scores)
            if cfg.merge :
                matrix_CEM.update_noise()
                scores_CEM = []
                weights_CEM = matrix_CEM.generate_weights(centroid,pop_size)
                for i in range(pop_size ):
                    workspace = Workspace()
                    w = weights_CEM[i]
                    torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())
                    eval_agent(workspace, t=0, stop_variable="env/done")
                    action = workspace["action"]
                    nb_steps += action[0].shape[0]
                    #stock the reward of each step in rewards
                    rewards = workspace["env/cumulated_reward"][-1]
                    #calculate the mean of all rewards 
                    mean_reward = rewards.mean()
                    logger.add_log("reward", mean_reward, nb_steps)
                    scores_CEM.append(mean_reward)
                #print("scores CEM ", scores_CEM)
                
                scores = scores + scores_CEM
                #print("scores ",scores)
                weights_CEMir = weights
                weights = weights + weights_CEM
                #print("weights_CEM ", weights_CEM)
                #print("weights_CEMir ", weights_CEMir)
                #print("weights ", weights)


            # Keep only best individuals to compute the new centroid
            elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
            elites_weights = [weights[k] for k in elites_idxs]
            #Concanetane the tensor of elites_weights
            elites_weights = torch.cat(
                [w.clone().detach().unsqueeze(0) for w in elites_weights], dim=0 #[torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
            )
            scores_elites.append([scores[i] for i in elites_idxs])
            centroid = elites_weights.mean(0)
            if cfg.EXPERIMENT:
                matrix.update_covariance_inverse_resize(elites_weights)
                cem_variant  =  'iResize' 
            elif cfg.CEMi :
                matrix.update_covariance_inverse(elites_weights)
                cem_variant = 'i' 
            elif cfg.merge :
                matrix.update_covariance_inverse_resize(elites_weights)
                matrix_CEM.update_covariance(elites_weights)
                cem_variant  =  'merged'
            else :
                matrix.update_covariance(elites_weights)
                cem_variant =  ''

            scores_elites_np = np.asarray(scores_elites)
            for i in range(len(scores_elites_np)):
                Y  =  scores_elites_np[i,:]
                X = [i for j in range(len(scores_elites_np[0]))]

                fonction_json(Y.tolist()) #ajoute les scores des elites au json


            if make_plot : 
                axs[0].scatter(X,Y, color=colors[run] , s=3)
        
        if make_plot:
            X = [i for i in range(len(scores_elites))]
            meanY =  [np.mean(y) for y in scores_elites]
            medianY =  [np.median(y) for y in scores_elites]
            axs[1].plot(X,meanY,marker='o', color=colors[run] )
            axs[2].plot(X,medianY,marker='o', color=colors[run])
    
    if make_plot:
        for g in range(3):
            axs[g].set_xlabel('Generations')
            axs[g].grid(True)
        axs[0].set_ylabel('Score of elites')
        axs[1].set_ylabel('Mean Score of elites')
        axs[2].set_ylabel('Median Score of elites')
        axs[0].grid(False)
        plt.tight_layout()
        name_file  = f"plot_output/plot_{cfg.gym_env.env_name}_CEM{cem_variant}_R{cfg.algorithm.nb_runs}G{cfg.algorithm.max_epochs}{'diagMat' if cfg.algorithm.diag_covMatrix else'fullMat'}.png"
        os.makedirs('./plot_output/')
        plt.savefig(name_file)
        print(f"Generated plot: '{name_file}' ")


@hydra.main(
    config_path="./configs/",
    config_name="cem_CartPole.yaml",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg, lambda *args: None)


if __name__ == "__main__":
    main()
