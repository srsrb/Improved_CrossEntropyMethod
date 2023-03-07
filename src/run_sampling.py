import cem
from bbrl_examples.algos.cem.cem import run_cem
import make_json


def sample_jsons(ls_cfgs,ls_envs_names,method,specification):
    """ L'unique fonction a executer pour faire des tests
    ls_cfgs : la liste des fichiers de configs  -----> prblms avec hydra que je n'ai pas eu le temps de resoudre
    ls_envs_names -> noms des environements que l'on voudra donner dans le json
    method : la methode utilisee (CEM? CEMi?)
    specification :  un commentaire qui figurera dans le titre du fichier json"""
    json  = make_json.JSON_Summary(ls_envs_names,method,specification)
    for i in range(len(ls_cfgs)):
        cfg  =  ls_cfgs[i]
        add_scores =  lambda ls : json.add_scores_to_environment(ls_envs_names[i],ls) #fonction qu'on passe a CEM que l'algo utilisera pour ajouter ses points. On a pas besoin de lui passer le json
        run_cem(cfg,add_scores) # trouver un moyen de lui passer le fichier config !!!!!!!
    
    json.end_sampling()
