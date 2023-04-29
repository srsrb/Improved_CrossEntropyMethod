import algorithm_copy

from turtle import update
import numpy as np
import gym
import os


import torch
import hydra
import matplotlib.pyplot as plt
import matplotlib.colors

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.loggers import Logger
from bbrl_examples.models.actors import DiscreteActor, ContinuousDeterministicActor
from bbrl_examples.models.envs import create_no_reset_env_agent

from bbrl.visu.visu_policies import plot_policy

from cem_actuel import *


@hydra.main(
    config_path="./configs/",
    config_name="cem_caRTpole.yaml",
)
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    kwargs  =  {'seed' : cfg.algorithm.seed  
                     cfg.algorithm.centroid 
                    max_epochs 
                    pop_size 
                    elites_nb 

    } #pas fini, flemem et pas riorite

    def initialisation(seed):
        torch.manual_seed(seed)
        eval_env_agent = create_no_reset_env_agent(cfg)

        eval_agent = create_CEM_agent(cfg, eval_env_agent)
        centroid = torch.nn.utils.parameters_to_vector(eval_agent.parameters())
        kwargs  = {'centroid' :  centroid,
                   'sigma':cfg.algorithm.sigma,
                   'noise_multiplier':cfg.algorithm.noise_multiplier
                   'max_epochs' : cfg.algorithm.max_epochs}
                    'elites_nb': cfg.algorithm.elites_nb,
        
        best_score = -np.inf
        nb_steps = 0
        scores_elites = []  # scores des elites a chaque epoch (generation)
    algorithm_copy.CEM(cfg, lambda *args: None)


if __name__ == "__main__":
    main()
