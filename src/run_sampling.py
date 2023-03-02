import hydra
import cem
from  cem_actuel import run_cem
import make_json


def sample_jsons(ls_cfg_paths,ls_envs_names,specification):
    """ Lance les simulations des fichiers de configurations de CEM(i) aux emplacements  `{ls_cfg_paths[i]}/{ls_envs_names[i]}` et enregistre les resultats dans le json nomme `eval_{specification}.json`
    
            - ls_cfg_paths  -> liste des dossiers dans lequel le fichier config de l'environnement `i` se trouve
    
            - ls_envs_names -> noms des fichiers configs en `.yaml` des environnements que l'on souhaite executer
    
            - specification ->  un commentaire qui figurera dans le titre du fichier json"""

    json  = make_json.JSON_Summary(ls_envs_names,specification)

    for i in range(len(ls_cfg_paths)):
        

        cfg_path  =  ls_cfg_paths[i]
        cfg_name = ls_envs_names[i]
        print(f"----->>Launching the simulation for the environment : ", cfg_name, "\n")
        @hydra.main(config_path=cfg_path,
        config_name = cfg_name)
        def run_it(cfg):
            add_scores =  lambda ls : json.add_scores_to_environment(ls_envs_names[i],ls) #fonction partielle qu'on passe a CEM que l'algo utilisera pour ajouter ses points. On a pas besoin de lui passer le json
            run_cem(cfg,add_scores)     
        run_it()
    json.end_sampling()


def sample_these_configs( ls_names,directory="./configs/",specification  = ""):
    """
    Fait la simulation des environements dont le nom des fichiers de configuration sont listes dans ls_names.
    Version plus pratique que sample_json

    Output le fichier json correspondant aux simulations
        - directory ->  chemin par defaut du dossier contenant les configs. L'argument est initialise par defaut a `./configs/`.

        - ls_names  -> liste de snoms des fichiers de configs utilises sans l'extension yaml
        - specification -> commentaire qui sera present dans le nom du fichier

    Exemple:
    Le dossier mon/dossier/favori contient trois fichiers de configuration que l'on souhaite lancer:
    ```md
    .
    └── .mon/dossier/favori
        ├── cem_cartpole_10runs.yaml
        ├── cemi_carpole_diag_noPlot_16runs.yaml
        └── cem_100gen_plot_3runs.yaml
    ```
    _

    ```python
    sample_these_configs(["cem_cartpole_10runs",
        "cemi_carpole_diag_noPlot_16runs",
        "cem_100gen_plot_3runs"],
        directory="./mon/dossier/favori",
        specification = "mon_commentaire"
        )
    ```
    Lancera l'evaluation des 3 environnements configures par les fichiers `.yaml` de la liste, situes a l'emplacement `./mon/dossier/favori`,
    et enregistrera les resultats sous le json :    `eval_mon_commentaire.json`.
    L'emplacement `.` correspond a l'emplacement du fichier du parametre directory correspond a l'emplacement de ce module (i.e. `run_sampling.py`)
    """
    ls_env_names =  [] 

    for name in ls_names:
        name_file  =  name+".yaml"
        ls_env_names.append( name_file)
    ls_cfg_paths = [directory for _ in range(len(ls_names))]
    print("ls cfg: ", ls_cfg_paths)

    sample_jsons(ls_cfg_paths,ls_env_names,specification)
    

sample_these_configs( ["cem_cartpole",'cem_cartpole2'])