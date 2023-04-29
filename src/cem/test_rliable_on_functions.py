import numpy as np
from rliable_generation import rliable_report as rly
from rliable_generation import JSON_Generator as json_gen
from rliable_generation import rliable_helpers as h
from complex_function.constant import *
from complex_function.algorithm_copy import *
import inspect
from complex_function import test_function

ls_funct = np.array(inspect.getmembers(
	test_function, inspect.isfunction))[:, 1]
print("Functions identified in module test_function:", ls_funct)
ls_algos = [CEM, CEMi, CEMir, CEM_plus_CEMi, CEM_plus_CEMir]


def make_jsons_for_all_functions(nb_runs):

	parameter_dict = {'centroid': torch.FloatTensor(
		centroid), 'seed': seed, 'sigma': sigma, 'noise_multiplier': noise_multiplier, 'max_epochs': max_epochs, 'pop_size': pop_size, 'elites_nb': elites_nb}
	print(parameter_dict['seed'])
	json = json_gen.JSON_Generator(
		ls_funct, ls_algos, nb_runs, **parameter_dict)

	json.generate_jsons()

#make_jsons_for_all_functions(5)

def get_min_maxs(ls_algos_names, folder_json):
	ls_algos_names = [algo.__name__ for algo in ls_algos]
	dict_json = [h.load_json(alg, folder_json) for alg in ls_algos_names]

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


mins, maxs = get_min_maxs(ls_algos, './FolderJson')
print(mins,maxs)


def make_rliable_analysis():
	ls_algos_names = [algo.__name__ for algo in ls_algos]

	analyzer = rly.rliable_Analyzer(ls_algos_names, './FolderJson', mins, maxs)
	#analyzer.plot_sample_efficiency_curve()
	#analyzer.plot_performance_profiles()
	analyzer.plot_aggregate_metrics()
	#ls_pairs = [ (a.__name__ , b.__name__) for a in ls_algos for b in ls_algos]
	#analyzer.plot_probability_improvement(ls_pairs)

make_rliable_analysis()
