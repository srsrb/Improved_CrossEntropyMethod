import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from convergence_analysis import *
from test_function import *
from bbrl_examples.algos.cem.complex_function.constant import *
import bbrl_examples.algos.cem.complex_function.algorithm_copy as alg
import torch
import os

score_function = Rastrigin


def plot_cem_on_function(f = score_function, force_CEM_version  = None,folder_plots="plots", folder_analysis="analysis",
						 show_plot=False, show_analysis=False,
						 show_points=False, show_ellipsoids=False, show_centroids=True):
	"""Plot an exectution of the CEM with `f` as the score function

	Parameters

	------

	`f` 
			function that takes a vector of `R^2` as an input an returns a real number

	folder_plots ="plots_"
			The directory in which the plot will be generated and saved

				ls_scores  =  [env(torch.from_numpy(w)).item() for w in all_centroids] # POUR ENVS GYM
	folder_analysis ="analysis_"
			The directory in which the analysis will be generated and saved


	show_plot=False
			Show the plot generated for the execution

	show_analysis=False
			Show the analysis generated for the execution


	`show_points (=False)`
			if `show_plot==True`
					if `show_points==True` then the plot draws the points corresponding to every every individuals.

	` show_ellipsoids=True` `show_centroids=True`
			Draw the ellipsoids

	show_centroids=True
			draw the centroids at each generation

	Side-Effects

	---

	Plots the execution (according to the parameters in the module `constant`) ans saves it  """

	np.random.rand(seed)
	cmap = plt.cm.rainbow
	norm = matplotlib.colors.Normalize(vmin=1, vmax=max_epochs)

	name  =  f.__name__
	centroid = centroids[name]

	parameter_dict = {'centroid': torch.FloatTensor(
		centroid), 'seed': seed, 'sigma': sigma, 'noise_multiplier': noise_multiplier, 'max_epochs': max_epochs, 'pop_size': pop_size, 'elites_nb': elites_nb, 'seuil_convergence': seuil_convergence[name], 'delta_convergence': delta_convergence[name]}
	
	algo_to_use = version_CEM if force_CEM_version == None else force_CEM_version
	match algo_to_use:
		case 'CEM':
			method = alg.CEM
		case 'CEMi':
			method = alg.CEMi
		case 'CEMir':
			method = alg.CEMir
		case 'CEM+CEMi':
			method = alg.CEM_plus_CEMi
		case 'CEM+CEMiR':
			method = alg.CEM_plus_CEMir
		case 'CEM_circle':
			method = alg.CEM_circle
		case _:
			raise Exception(
				"Invalid version of CEM in the module constant.py: " + version_CEM)

	results  =  method(f,**parameter_dict)
	
	all_weights = results['all_weights']
	all_weights = results['all_weights']
	all_elites= results['all_elites'] 
	all_elite_scores=  results['all_elite_scores']
	all_centroids= results['all_centroids']
	all_covs= results['all_covs']
	all_percentages= results['all_percentages']
	list_matrix_names = results['list_matrix_names']
	convergence_generation =  results['convergence_generation']


	x_min, y_min = np.min(all_weights[:, :, 0]), np.min(all_weights[:, :, 1])
	x_max, y_max = np.max(all_weights[:, :, 0]), np.max(all_weights[:, :, 1])

	x = np.linspace(x_min-1, x_max+1, num=200)
	y = np.linspace(y_min-1, y_max+1, num=200)
	X, Y = np.meshgrid(x, y)
	Z = f([X, Y])
	plt.figure(figsize=(12, 12))
	plt.contourf(X, Y, Z, pop_size, cmap='viridis')
	plt.colorbar()
	plt.grid()
	ax = plt.gca()
	cmap = plt.cm.rainbow

	N = len(all_weights)
	col_ellipse = matplotlib.colors.Normalize(vmin=0, vmax=N)

	for mat_i in range(N):
		weights = all_weights[mat_i]
		covs = all_covs[mat_i]
		assert len(
			weights) == max_epochs,  "Error, max_epochs != number of generations simulated"

		for generation_i in range(max_epochs):
			weights_gen_i = weights[generation_i]
			if show_points:
				ax.scatter(weights_gen_i[:, 0], weights_gen_i[:, 1], color=cmap(
					norm(generation_i)), marker='o', s=3)
			if show_ellipsoids:
				edgecolor = cmap(col_ellipse(mat_i))
				alg.plot_confidence_ellipse(
					covs[generation_i], ax, all_centroids[generation_i], n_std=1., edgecolor=edgecolor)
	if show_centroids:
		ax.plot(all_centroids[:, 0], all_centroids[:, 1], marker='^', c='r')

	plt_title = f" {f.__name__} function - {algo_to_use} - {max_epochs} generations (starting point = {centroid}) "
	plt.title(plt_title)
	file_name = f"{f.__name__}/{algo_to_use}_{max_epochs}G_starting_point{centroid}) "
	if not os.path.exists(f'./{folder_plots}/{f.__name__}'):
		os.makedirs(f'{folder_plots}/{f.__name__}')
	plt.savefig(folder_plots+'/'+file_name+".png")
	if show_plot:
		plt.show()

	param_analyzer = {'score_function': f, 'all_labels': list_matrix_names, 'all_percentages': all_percentages, 'all_weights': all_weights,
					  'all_covs': all_covs, 'all_centroids': all_centroids, 'folder': folder_analysis, 'file_name': plt_title}

	analyzer = convergence_Analyzer(**param_analyzer)
	analyzer.xy_centroid_evolution()
	analyzer.variance_evolution()
	analyzer.comparison_slope_vs_ellipsoid_axis()
	analyzer.percentage_evolution()


def plot_every_function(**kwargs):
	"""Generate a plot for every function in the module `test_function` along with an Analyse for every execution

	Parameters
	---
	 folder_plots ="plots_"
			The directory in which the plots will be generated and saved

	folder_analysis ="analysis_"
			The directory in which the analysis will be generated and saved

	show_plot=False
			Show the plot generated for the execution

	show_analysis=False
			Show the analysis generated for the execution


	`show_points (=False)`
			if `show_plot==True`
					if `show_points==True` then the plot draws the points corresponding to every every individuals.

	` show_ellipsoids=True` `show_centroids=True`
			Draw the ellipsoids

	show_centroids=True
			draw the centroids at each generation

	Side-Effects

	---

	Plots the execution (according to the parameters in the module `constant`) ans saves it 

	"""
	import inspect
	import test_function
	ls_funct = np.array(inspect.getmembers(
		test_function, inspect.isfunction))[:, 1]
	for f in ls_funct:
		print(f)
		plot_cem_on_function(f=f, **kwargs)


def plot_CEM_every_algo_generique(plot_function,**kwargs):

	for cem_name in ls_versions_CEM:
		kwargs['force_CEM_version'] = cem_name
		plot_function(**kwargs)

def plot_CEM_every_algo_one_function(f, **kwargs):
	plot_CEM_every_algo_generique(plot_cem_on_function,**{'f':f,**kwargs})

def plot_CEM_every_algo_every_function( **kwargs):
	plot_CEM_every_algo_generique(plot_every_function,**kwargs)


if __name__ == '__main__':
	
	kwargs  = {}
	kwargs.update( force_CEM_version  = "CEMi",folder_plots="plots", folder_analysis="analysis",
						 show_plot=False, show_analysis=False,
						 show_points=True, show_ellipsoids=True, show_centroids=True)
	plot_CEM_every_algo_one_function(Sphere, **kwargs)
	plot_CEM_every_algo_one_function(mishra_bird, **kwargs)
	plot_CEM_every_algo_one_function(Griewank, **kwargs)