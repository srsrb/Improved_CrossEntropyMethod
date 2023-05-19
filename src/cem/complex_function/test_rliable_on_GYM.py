from bbrl.workspace import Workspace
from bbrl_examples.models.envs import create_no_reset_env_agent
import numpy as np
import hydra
from omegaconf import OmegaConf as oc
from bbrl_examples.algos.cem.rliable_generation import rliable_report as rly
from bbrl_examples.algos.cem.rliable_generation import JSON_Generator as json_gen
from bbrl_examples.algos.cem.rliable_generation import rliable_helpers as h
from bbrl_examples.algos.cem.complex_function.cem_actuel import create_CEM_agent
from bbrl_examples.algos.cem.complex_function.constant import *
from bbrl_examples.algos.cem.complex_function.algorithm_copy import *
import inspect
from bbrl_examples.algos.cem.complex_function import test_function
from bbrl_examples.algos.cem.complex_function.algorithm_copy import *

import copy

ls_algos = [CEM, CEMi, CEM_plus_CEMi]

def rename(newname):
	def decorator(f):
		f.__name__ = newname
		return f
	return decorator

def make_jsons_for_all_GYM_env(nb_runs,lst_GYM_configs):
	
	ls_envs = []
	lst_parameter_dict = []
	for c in lst_GYM_configs:
		
		cfg = oc.load("./configs/"+c)

		eval_env_agent = create_no_reset_env_agent(cfg)
		eval_agent = create_CEM_agent(cfg, eval_env_agent)
		

		centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
		
		# -------------------------------------------------------------------------------------
		# A modifier : c -> cfg.gym_env.env_name
		@rename(c)
		def score_function(w,eval_agent=eval_agent):
			workspace = Workspace()
			torch.nn.utils.vector_to_parameters(w, eval_agent.parameters())
			eval_agent(workspace, t=0, stop_variable="env/done")
			rewards = workspace["env/cumulated_reward"][-1]
			mean_reward = rewards.mean()

			return -mean_reward
		
		
		conf = cfg.algorithm
		
		parameter_dict = {'seuil_convergence': conf.seuil_convergence, 'delta_convergence': conf.delta_convergence,'centroid': torch.FloatTensor(
		centroid), 'seed': conf.seed, 'sigma': conf.sigma, 'noise_multiplier': conf.noise_multiplier, 'max_epochs': conf.max_epochs, 'pop_size': conf.pop_size, 'elites_nb': conf.elites_nb}

		ls_envs.append(score_function)
		lst_parameter_dict.append(parameter_dict)
	
	
	json = json_gen.JSON_Generator(
		ls_envs, ls_algos, nb_runs, lst_parameter_dict)
		
	json.generate_jsons()

# make_jsons_for_all_GYM_env(4, ["cem_cartpole.yaml","cem_acrobot.yaml","cem_mountaincar.yaml","cem_pendulum.yaml"])


def get_min_maxs(ls_algos, folder_json):
	"""
	# Input
		liste des algorithmes utilises
		folder des fichiers jsons

	# OUtput
		2 dicts avec les mins e t les maxs respectifs des differents algos dans folder_JSON
	"""
	ls_algos = [algo.__name__ for algo in ls_algos]
	dict_json = [h.load_json(alg, folder_json) for alg in ls_algos]

	ls_env = list(dict_json[0].keys())

	dict_mini = {}
	dict_maxi = {}

	for env in ls_env:
		ls_env_vals = []
		for dict in dict_json:
			ls = np.asarray(dict[env])
			ls_env_vals.append(ls)
		ls_env_vals = np.asarray(ls_env_vals).flatten()
		dict_mini[env] = np.min(ls_env_vals)
		dict_maxi[env] = np.max(ls_env_vals)
	return dict_mini, dict_maxi

def make_rliable_analysis():
	ls_algos_names = [algo.__name__ for algo in ls_algos]
	mins, maxs = get_min_maxs(ls_algos, './FolderJson')
	analyzer = rly.rliable_Analyzer(ls_algos_names, './FolderJson', mins, maxs)
	analyzer.plot_sample_efficiency_curve()
	analyzer.plot_performance_profiles()
	analyzer.plot_aggregate_metrics()
	ls_pairs = [ ('CEM','CEMi') , ('CEM','CEM_plus_CEMi'), ('CEMi','CEM_plus_CEMi')]
	analyzer.plot_probability_improvement(ls_pairs)

make_rliable_analysis()