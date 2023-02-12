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
from bbrl_examples.models.actors import DiscreteActor ,DiscreteDeterministicActor, ContinuousDeterministicActor
from bbrl_examples.models.envs import create_no_reset_env_agent

from bbrl.visu.visu_policies import plot_policy


class CovMatrix:
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier):
        #le nombre de paramétres
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):

        policy_dim = centroid.size()[0]
        if False:
          # Separable CEM, useful when self.policy_dim >> 100
          # Use only diagonal of the covariance matrix
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
    
    def update_covariance_inverse(self, elite_weights) -> None:
        cov = torch.cov(elite_weights.T)+ self.noise
        u  =  torch.linalg.cholesky(cov)

        inv =  torch.cholesky_inverse(u)
       # print("Ma Matrice next mat de  cov?\: \n ", inv)
       # print("Produit?", torch.matmul(cov,inv))
       # print("Eigen: ", torch.linalg.eigvals(inv))
        
        self.cov = inv


# Create the PPO Agent
def create_CEM_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    if isinstance(env_agent.action_space , gym.spaces.Discrete):
        action_agent = DiscreteActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
    elif isinstance(env_agent.action_space, gym.spaces.Box):
        action_agent = DiscreteDeterministicActor(
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

    #j'ai pas compris comment on a initialiser centroid (que représentent eval_agent.parameters()) 
    centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
    
    matrix = CovMatrix(
        centroid,
        cfg.algorithm.sigma,
        cfg.algorithm.noise_multiplier,
    )
    #print(matrix.cov)
    best_score = -np.inf
    nb_steps = 0

    #creation d'un fichier pour stocker les résultats 
    f = open("cem_cartpole_100.txt", "w")
    scores_elites = [] #scores des elites a chaque epoch (generation)
    # 7) Training loop
    for epoch in range(30):
        print("Step ", epoch)

        matrix.update_noise()
        scores = []
        weights = matrix.generate_weights(centroid, pop_size)

        for i in range(pop_size):
            workspace = Workspace()
            w = weights[i]
            torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())

            eval_agent(workspace, t=0, stop_variable="env/done")
            action = workspace["action"]
            nb_steps += action[0].shape[0]
            rewards = workspace["env/cumulated_reward"][-1]
            mean_reward = rewards.mean()
            logger.add_log("reward", mean_reward, nb_steps)

            # ---------------------------------------------------
            scores.append(mean_reward)
            """
            if cfg.verbose:
                #f.write(f"Indiv: {i + 1} score {scores[i]:.2f} nb_steps: {nb_steps}, reward: {mean_reward}\n")
                #print(f"Indiv: {i + 1} score {scores[i]:.2f}")
                #print(f"nb_steps: {nb_steps}, reward: {mean_reward}")
            if cfg.save_best and mean_reward > best_score:
                best_score = mean_reward
                print("Best score: ", best_score)
                directory = "./cem_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#cem_basic#team#"
                    + str(mean_reward.item())
                    + ".agt"
                )
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        "./cem_plots/",
                        cfg.gym_env.env_name,
                        best_score,
                        stochastic=False,
                    )
                """
        # Keep only best individuals to compute the new centroid
        elites_idxs = np.argsort(scores)[-cfg.algorithm.elites_nb :]
        elites_weights = [weights[k] for k in elites_idxs]
        elites_weights = torch.cat(
            [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
        )
        
        scores_elites.append([scores[i] for i in elites_idxs])
        centroid = elites_weights.mean(0)
        # Update covariance
        matrix.update_noise()
        matrix.update_covariance_inverse(elites_weights)
    f.close()

    #----------------Plotting
    scores_elites = np.asarray(scores_elites)
    X = [i for i in range(len(scores_elites))]
    for i in range(len(scores_elites[0])):
        Y  =  scores_elites[:,i]
        plt.scatter(X,Y)

    medianY =  [np.median(y) for y in scores_elites]
    meanY =  [np.mean(y) for y in scores_elites] 
    plt.plot(X,medianY,marker='x', label= "medianScore")
    plt.plot(X,meanY,marker='o', label = "meanScore" )
    
    plt.title('Evolution of the Elite Scores - iCEM')
    plt.xlabel('Generation')
    plt.ylabel('Score of elites')
    plt.legend()
    
    plt.text(0, 400, f'#elites_elected= {cfg.algorithm.elites_nb} , nnSize:{cfg.algorithm.architecture.actor_hidden_size} ' + r'$\Sigma = $' + str(cfg.algorithm.sigma)  )
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
