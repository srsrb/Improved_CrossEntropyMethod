import json
import os
from bbrl_examples.algos.cem.complex_function.constant import *
from bbrl_examples.algos.cem.complex_function import cem_actuel
import numpy as np
import torch

class JSON_Object:
	"""JSON object that can store the result of multiple runs of an algorithm on different environments.
	"""

	def __init__(self,ls_envs_names ,  file_name, folder = "./FolderJson" ) -> None:
		"""Create a new JSON object that can store the result of multiple runs of an algorithm on different environments.
		
		Parameters
		--
		ls_envs 
			list of the environments anmes tested \n
		file_name 
		 	name of the output JSON file\n
		"""
		self.ls_envs = ls_envs_names
		self.dict = {}
		self.folder = folder
		self.file_name = file_name

		for env in self.ls_envs:
			self.dict[env] = []
	
	def add_run_to_environment(self,env,run: list[list[int]]) -> None:
		"""Adds the run to the corresponding environment env to the sampling

		Parameters
		----------
		env : 'str'
			name of the Environment the run corresponds to

		run : list[list[int]]
			Scores corresponding to the current run:

			run  =  [ ls_scores_for_generation_1, ls_scores_for_generation_2, ... ls_scores_for_generation_G ]

			with ls_scores_for_generation_i  =  [score_w_1, ... score_w_PopulationSize, ]

		Raises
		------
		Exception
		   "Environement {env}  is not tested in this simulation, since this JSON has not been initialized with it.\n \
			Here are the environments evaluated : {self.ls_envs}."
			
		"""
		if env in self.dict:
			self.dict[env].append(run)        
		else:
			err  = f"Environement {env}  is not tested in this simulation, since this JSON has not been initialized with it.\n \
			Here are the environments evaluated : {self.ls_envs}."
			raise Exception(err)
	
	def _check_completeness_of_evaluation(self):
		"""Checks if all the environments have been evaluated."""
		for env in self.dict:
			if (self.dict[env] == []):
				raise Exception( f"No sample for the environment {env} \n")
	
	def end_sampling(self):
		"""End sampling and generate json.
		Will generate a JSON file usable by the rliable library, named "eval_{specification} " """
		self._check_completeness_of_evaluation()
		
		nb_runs  =  len(self.dict[self.ls_envs[0]])
		print(f"Sampled {len(self.ls_envs)} environments with {nb_runs} runs for {self.file_name}.json ! \n ")

		if not os.path.exists(f'./{self.folder}'):
			os.makedirs(f'./{self.folder}')
		out_file = open(f'./{self.folder}/{self.file_name}.json','w+')
		json.dump(self.dict,out_file)
		print(f'Generated \"{self.folder}/{self.file_name}.json\" !')

class _JSON_Generator_Single_Algorithm():
	"""generates the JSON file for a certin number of run for the specified algoritgm"""

	def __init__(self , ls_environments_f, algorithm_f, nb_runs, lst_parameter_dicts, seed  =  42) -> None:
		"""generates the JSON file for a certin number of run for the specified algoritgm

		Parameters
		----------
		ls_environments_f : _type_
			list of the environments functions (aka score functions) that are to be used for the different executions of the specified algorithm
		algorithm_f : _type_
			the algorithm tested
		nb_runs : _type_
			The number of independant runs
		kwargs:
			kwargs parameters for `algorithm_f` 
		"""
		self.ls_environments = ls_environments_f # list of score functions
		self.algo = algorithm_f
		self.nb_runs =  nb_runs
		self.seed  = seed
		
		self.lst_parameter_dicts = lst_parameter_dicts

		ls_envs_names  =   [ env.__name__ for env in ls_environments_f ]

		self.JSON_object = JSON_Object(ls_envs_names, algorithm_f.__name__)

	def _merge_weights(self,all_weights):
		"""merge the weights obtained with the different matrices as if only one was used"""
		weights_merged = []

		N_generations  =  len(all_weights[0])
		for i in range(N_generations):
			ls_gen  =  []
			for mat in all_weights:
				gen_i =  mat[i]
				ls_gen+= gen_i
			
			weights_merged.append(ls_gen)
		
		return weights_merged

	def _execute_runs(self):
		seed0 = self.seed

		for i in range(self.nb_runs):
			print(f"Run {i}/{self.nb_runs} with {self.algo.__name__}")
		
			seed = seed0+i
			
			for i in range(len(self.ls_environments)):
				env  =  self.ls_environments[i]
				param_dict  =  self.lst_parameter_dicts[i]
				param_dict['seed'] = seed
				all_centroids = self.algo(env,**param_dict)['all_centroids']
				
				# all_centroids = cem_actuel.isoler_resultat(self.algo(env,**self.kwargs),'all_centroids')

				# all_centroids = all_centroids.tolist()
				ls_scores  =  [env(torch.from_numpy(w)).item() for w in all_centroids] # POUR ENVS GYM
				# ls_scores  =  [env(w) for w in all_centroids] # POUR FCTS 2D
				self.JSON_object.add_run_to_environment(env.__name__,ls_scores)
	
	def generate_json(self):
		self._execute_runs()
		self.JSON_object.end_sampling()

class JSON_Generator():
	"""generates the JSON file for a certin number of run for the specified algorithms"""
	
	def __init__(self , ls_environments_f, ls_algorithms_f, nb_runs, lst_parameter_dicts):
		"""generates the JSON file for a certin number of run for the specified algorithms

		Parameters
		----------
		ls_environments_f : _type_
			list of the environments functions (aka score functions) that are to be used for the different executions of the specified algorithm
		algorithm_f : _type_
			the algorithm tested
		nb_runs : _type_
			The number of independant runs
		
		"""
		self.ls_JSONS = [ _JSON_Generator_Single_Algorithm(ls_environments_f,algo,nb_runs,lst_parameter_dicts) for algo in ls_algorithms_f]

	def generate_jsons(self):
		for JSON_obj in self.ls_JSONS:
			JSON_obj.generate_json()

	