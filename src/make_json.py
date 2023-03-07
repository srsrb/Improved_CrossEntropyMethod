import json
class JSON_Summary:
    
    def __init__(self,ls_envs , method, specification) -> None:
        """initialize attributes
        ls_envs -> list of the environments tested \n
        method -> name of method used \n 
        specification -> additional specification that will apppear in the name of the file\n"""
        self.ls_envs = ls_envs
        self.dict = {}
        self.method  = method
        self.specification  =  specification

        for env in self.ls_envs:
            self.dict[env] = []
    
    def add_scores_to_environment(self,env,ls_scores) -> None:
        """adds the scores in ls_scores to the environment env"""
        if env in self.dict:
            self.dict[env] = self.dict[env]+ls_scores
            print("Added ", len(ls_scores), " samples to ", env)
        else:
            err  = f"Environement {env}  is not tested in this summary, since this JSON has not been initialized with it.\n \
            Here are the environments evaluated : {self.ls_envs}."
            raise Exception(err)
    
    def check_completeness_of_evaluation(self):
        """Checks if all the environments have been evaluated."""
        for env in self.dict:
            if (self.dict[env] == []):
                raise Exception( f"No sample for the environment {env} \n")
    
    def end_sampling(self):
        """End sampling and generate json.
        Will generate a JSON file usable by the rliable library, named "eval_{name_of_method}_{specification} " """
        self.check_completeness_of_evaluation()
        print(f"Sampled {len(self.ls_envs)} environments: \n ")
        

        out_file = open(f'eval_{self.method}_{self.specification}.json','w+')
        json.dump(self.dict,out_file)
        print(f'Generated \"eval_{self.method}_{self.specification}\" !')


def test_json():
    sum  =  JSON_Summary(['tu_craques'],"demo","commentaire")
    sum.add_scores_to_environment('tu_craques',[20.37, 31.7, 15.979999, 25.85, 21.12, 15.35])
    sum.end_sampling()
test_json()