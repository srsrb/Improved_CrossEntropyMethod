import numpy as np
from rliable_generation import rliable_report as rly
from rliable_generation import JSON_Generator as json_gen
from complex_function.constant import *
from complex_function.algorithm_copy import *
import inspect
from complex_function  import test_function

mins  =  {"Ackley": -500,"Griewank": -500,"Holder": -500,"Rastrigin": -500,"Rosenbrock": -500,"Sphere": -500,"booth": -500,"distance": -500,"mishra_bird": -500}

maxs = {"Ackley": 1,"Griewank": 1,"Holder": 1,"Rastrigin": 1,"Rosenbrock": 1,"Sphere": 1,"booth": 1,"distance": 1,"mishra_bird": 1}

ls_funct = np.array(inspect.getmembers(test_function,inspect.isfunction))[:,1]
print("Functions identified in module test_function:",ls_funct)
ls_algos  =  [CEM,CEMi,CEMir,CEM_plus_CEMi,CEM_plus_CEMir]

def make_jsons_for_all_functions(nb_runs):

	parameter_dict = {'centroid': torch.FloatTensor(
        centroid), 'seed': seed, 'sigma': sigma, 'noise_multiplier': noise_multiplier, 'max_epochs': max_epochs, 'pop_size': pop_size, 'elites_nb': elites_nb}

	json = json_gen.JSON_Generator(ls_funct,ls_algos,nb_runs,**parameter_dict)

	json.generate_jsons()

def make_rliable_analysis():
	ls_algos_names = [algo.__name__ for algo in ls_algos]

	analyzer  = rly.rliable_Analyzer(ls_algos_names,'./FolderJson',mins,maxs)
	analyzer.plot_sample_efficiency_curve()

make_rliable_analysis()