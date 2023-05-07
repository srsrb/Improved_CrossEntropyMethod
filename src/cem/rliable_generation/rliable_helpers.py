from matplotlib import rc
from matplotlib import rcParams
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import logging
from rliable import library as rly
from rliable import metrics
# @title Imports

import numpy as np

import json
import os.path as osp


# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# @title Plotting: Seaborn style and matplotlib params

sns.set_style("white")

# Matplotlib params

rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

rc('text', usetex=False)

arr = np.asarray


def set_axes(ax, xlim, ylim, xlabel, ylabel):
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		ax.set_xlabel(xlabel, labelpad=14)
		ax.set_ylabel(ylabel, labelpad=14)

def set_ticks(ax, xticks, xticklabels, yticks, yticklabels):
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticklabels)

def decorate_axis(ax, wrect=10, hrect=10, labelsize='large'):
		# Hide the right and top spines
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_linewidth(2)
		ax.spines['bottom'].set_linewidth(2)
		# Deal with ticks and the blank space at the origin
		ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
		# Pablos' comment
		ax.spines['left'].set_position(('outward', hrect))
		ax.spines['bottom'].set_position(('outward', wrect))

# @title Helpers for normalizing scores and plotting histogram plots.

def normalize_ls(list : list[float], mini :float, maxi : float):
		
		"Normalized the list with respect to mini and maxi"
		return [ (x- mini)/(maxi-mini) for x in list]

def convert_to_matrix(dict):
		keys = sorted(list(dict.keys()))
		return np.stack([dict[k] for k in keys], axis=1) # ( nb_run * nb_environments)

def plot_score_hist(score_matrix, names, bins=20, figsize=(28, 14),
										fontsize='xx-large', N=6, extra_row=1):
		"""
		names  :  names of the environments"""
		num_tasks = score_matrix.shape[1]

		N1 = (num_tasks // N) + extra_row
		fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
		for i in range(N):
				for j in range(N1):
						idx = j * N + i
						if idx < num_tasks:
								ax[j, i].set_title(names[idx], fontsize=fontsize)
								sns.histplot(score_matrix[:, idx],
														 bins=bins, ax=ax[j, i], kde=True) # type: ignore
						else:
								ax[j, i].axis('off')
						decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
						ax[j, i].xaxis.set_major_locator(MaxNLocator(4))
						if idx % N == 0:
								ax[j, i].set_ylabel('Count', size=fontsize)
						else:
								ax[j, i].yaxis.label.set_visible(False)
						ax[j, i].grid(axis='y', alpha=0.1)
		return fig

# @title Stratified Bootstrap CIs and Aggregate metrics

StratifiedBootstrap = rly.StratifiedBootstrap

def IQM(x): return metrics.aggregate_iqm(x)  # Interquartile Mean
def OG(x): return metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
def MEAN(x): return metrics.aggregate_mean(x)
def MEDIAN(x): return metrics.aggregate_median(x)

def load_json(file_name, base_path):
		"""Creates a dictionnary associated with the json file : `base_path/file_name.json`

		Parameters
		----------
		file_name: str
										Name of the json file to be loaded as a dict 
		base_path :  str
										path to the json file

		Returns
		-------
		dict
										A dict corresponding to the json file
		"""
		path = osp.join(base_path, f'{file_name}.json')
		with open(path, 'r') as f:
				scores = json.load(f)
		scores = {game: val for game, val in scores.items()}
		return scores

def _create_normalized_samples (dict_samples: dict[str,list[list[float]]], min_scores: dict[str,float], max_scores: dict[str,float]):
		"""Normalizes each score fo the input dictionnary for every environment according to the mins and maxs specified in min_scores and max_scores .

		(/!\\ Disambiguation :  Difference between  teh samples and scores associated to a run :
																																								- A run is one execution with a particular seed of an algorithm
										- The score of a run is the value attributed  to the centroid by the score function at the last generation of the run
										- The sample associated to a run  is the set of all the scores obtained by each centroid at each generation, for 1 run ( a 1D list)
		)

		THE INPUT DICTIONNARY MUST HAVE THE CORRECT FORMATING

		Parameters
		----------
		base_path : str
										path of the folder containing the json

		min_scores : dict[str,list[int]]
										Dict containing the lists of the different minimas for every sample of each game :
										 >>> {
														env1 : min_env1,
														env2: min_env2
										}
										Used for the normalization of the scores

		max_scores : dict[str,list[int]]
										Dict containing the lists of the different minimas for every sample of each game :
										 >>> {
														env1 : max_env1,
														env2:max_env2
										}

		Returns
		---
		A dict of all the samples but normalized with the provided maxs and mins
		"""
		
		normalized_samples = {}
		for env, ls_run in dict_samples.items():
			ls  =  [normalize_ls(run, min_scores[env], max_scores[env])	 for run in ls_run]
			normalized_samples[env] = ls
		
		return normalized_samples

def _create_normalized_scores(dict_samples: dict[str,list[list[float]]], min_scores: dict[str,float], max_scores: dict[str,float]):
	"""Creates a dict of normalized scores from a dict of normalized samples (generated by the function `create_normalized_samples`), along with the corresponding score matrix ( of size nb_runs * nb_games)

	Parameters
	----------
	normalized_samples : dict[str,list[list[int]]]
					Normalized samples

	Returns
	-------
	tuple [ dict[str,list[int]] ,  score_matrix ]

	"""
	
	normalized_samples = _create_normalized_samples(dict_samples,min_scores,max_scores)

	normalized_scores =  {}
	for env,ls_runs in normalized_samples.items():
		normalized_scores[env] = [ run[-1] for run in ls_runs ] 
	
	return normalized_scores

def create_score_matrix(dict_samples,min_scores,max_scores):
	normalized_scores = _create_normalized_scores(dict_samples,min_scores,max_scores)
	score_matrix = convert_to_matrix(normalized_scores)
	return score_matrix

def create_score_matrix_all_gen(dict_samples: dict[str,list[list[float]]], min_scores: dict[str,float], max_scores: dict[str,float]):
	
	normalized_samples = _create_normalized_samples(dict_samples,min_scores,max_scores)
	
	score_matrix = convert_to_matrix(normalized_samples)
	return score_matrix

def zip_to_dictionnary(X,Y):
	return dict(zip(X,Y))

def make_procgen_pairs(ls_pairs,score_matrix_dict):
	return { u+','+v: (score_matrix_dict[u],score_matrix_dict[v]) for u,v in ls_pairs}