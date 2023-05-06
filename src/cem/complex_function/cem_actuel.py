from turtle import update
import numpy as np
import gym
import os

from omegaconf import OmegaConf as omg

import torch
import torch.nn as nn
import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.loggers import Logger
from bbrl_examples.models.actors import DiscreteActor, ContinuousDeterministicActor
from bbrl_examples.models.envs import create_no_reset_env_agent

from bbrl.visu.visu_policies import plot_policy

import bbrl_examples.algos.cem.complex_function.algorithm_copy as alg

# Create the PPO Agent


def create_CEM_agent(cfg, env_agent):
	obs_size, act_size = env_agent.get_obs_and_actions_sizes()
	if isinstance(env_agent.action_space, gym.spaces.Discrete):
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

def CEM_GYM_Generique(cfg, CEM_function):
	seed = cfg.algorithm.seed
	nb_runs = cfg.algorithm.nb_runs
	torch.manual_seed(cfg.algorithm.seed)

	assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

	dico = omg.to_container(cfg)
	kwargs = dico['algorithm']
	results = []
	for run in range(nb_runs):
		print(f'Run{run}/{nb_runs}')
		
		torch.manual_seed(seed)
		eval_env_agent = create_no_reset_env_agent(cfg)

		eval_agent = create_CEM_agent(cfg, eval_env_agent)

		centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())

		def score_function(w, seed):
			workspace = Workspace()
			torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())
			eval_agent(workspace, t=0, stop_variable="env/done")
			rewards = workspace["env/cumulated_reward"][-1]
			mean_reward = rewards.mean()

			return -mean_reward

		def f(w): return score_function(w, seed)
		results.append(CEM_function(f, **{'centroid': centroid, **kwargs}))
	return results

def CEM_GYM(cfg): return CEM_GYM_Generique(cfg, alg.CEM)

def CEMi_GYM(cfg): return CEM_GYM_Generique(cfg, alg.CEMi)

def CEMir_GYM(cfg): return CEM_GYM_Generique(cfg, alg.CEMir)

def CEM_GYM_circle(cfg) : return CEM_GYM_Generique(cfg,alg.CEM_circle)

def CEM_plus_CEMi_GYM(cfg): return CEM_GYM_Generique(cfg, alg.CEM_plus_CEMi)

def CEM_plus_CEMir_GYM(cfg): return CEM_GYM_Generique(cfg, alg.CEM_plus_CEMir)

def CEM_GYM_actuel(cfg):
	name = cfg.algorithm.name_version_cem
	lst_CEM_disponibles = ['CEM', 'CEMi', 'CEMir','CEM_circle','CEM+CEMi', 'CEM+CEMiR']
	match name:
		case 'CEM': f = CEM_GYM
		case 'CEMi': f = CEMi_GYM
		case 'CEMir': f = CEMir_GYM
		case 'CEM_circle': f = CEM_GYM_circle
		case 'CEM+CEMi': f = CEM_plus_CEMi_GYM
		case 'CEM+CEMir': f = CEM_plus_CEMir_GYM
		case _: raise Exception(f"Nom de CEM inconnu: {name}... N'appartient pas a {lst_CEM_disponibles}")
	return f(cfg)

def isoler_resultat(results, nom_resultat):
	"""retourne a la liste du resultat nom_resultat( all_weights, all_covs) sur chaque run de results"""

	ls = []
	for run in results:
		assert nom_resultat in run,  f"'{nom_resultat}' n'est pas un resultat dans le dictionnaire des resultats: {list(run.keys())}"
		ls.append(run[nom_resultat])
	return np.array(ls)

def make_plot_title(cfg):
	title = f""" Elite's score on several runs-  {cfg.algorithm.name_version_cem} :: architechture {cfg.algorithm.architecture.actor_hidden_size} """
	return title

def make_plot_filename(cfg):
	name   = f"{cfg.algorithm.name_version_cem}_{cfg.algorithm.nb_runs}runs.png"
	return name

def plot_elite_scores(name, resultats):
	all_runs_elite_scores = isoler_resultat(resultats, 'all_elite_scores')
	all_convergences = isoler_resultat(resultats, 'convergence_generation')
	fig, axs = plt.subplots(1, 3, figsize=(12, 5))
	fig.suptitle(name)
	fig.set_tight_layout(True)

	axs[0].set_xlabel('generation')
	axs[1].set_xlabel('generation')
	axs[2].set_xlabel('Generation')
	axs[0].set_ylabel('Score')
	axs[1].set_ylabel('Median Score')
	axs[2].set_ylabel('Mean Score')
	axs[0].grid()
	axs[1].grid()
	axs[2].grid()

	N_runs = np.shape(all_runs_elite_scores)[0]
	N_gen = np.shape(all_runs_elite_scores)[1]
	N_elites = np.shape(all_runs_elite_scores)[2]

	cmap = plt.cm.rainbow
	i_cmap = np.linspace(0, 1, N_runs)

	range_generations = [i for i in range(N_gen)]
	for run in range(N_runs):
		all_elite_scores = all_runs_elite_scores[run]
		color = cmap(i_cmap[run])

		all_medians = [np.median(gen) for gen in all_elite_scores]

		all_means = [np.mean(gen) for gen in all_elite_scores]
		for n in range(N_gen):
			elites_gen_n = all_elite_scores[n]
			axs[0].scatter(np.repeat(n, N_elites), elites_gen_n,
						   color=color, marker='o', s=3)
		axs[1].plot(range_generations, all_medians,
					color=color, marker='o')
		axs[2].plot(range_generations, all_means,
					color=color, marker='o', label=f'convergence: {all_convergences[run]}')

	fig.legend()

	return fig

def save_plot(name,fig,show = False):
	fig.savefig(name)
	if show:
		fig.show()
	print(f"Plot{name} saved! ^-^'")


@hydra.main(
	config_path="./configs/",
	config_name="cem_acrobot.yaml",
)
def main(cfg):
	import torch.multiprocessing as mp

	mp.set_start_method("spawn")
	resultats = CEM_GYM_actuel(cfg)
	title,  name= make_plot_title(cfg),make_plot_filename(cfg)
	fig  = plot_elite_scores(title, resultats)
	save_plot(name, fig)
	

if __name__ == "__main__":
	main()
