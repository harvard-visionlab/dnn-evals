import wandb
from .model_eval import ModelEval
from .utils._wandb import *

class ModelEvalWandb(ModelEval):           
    def scoreboard(self, model, transform):
        raise NotImplementedError("Subclasses of ModelEvalWandb should implement `scoreboard`.")        
    
    @property
    def wandb_filters(self):
        return {f"config.{k}": v for k,v in self.config.items()}
    
    def get_wandb_api(self, entity=None, project=None):
        if entity is None: entity = self.entity
        if project is None: project = self.project
        assert entity is not None, "Argument `entity` required for wandb logging"
        assert project is not None, "Argument `project` required for wandb logging"
        api = wandb.Api()
        
        return api,entity,project
    
    def log_tables(self, results, config, tags=[], entity=None, project=None):
        if entity is None: entity = self.entity
        if project is None: project = self.project
        assert entity is not None, "Argument `entity` required for wandb logging"
        assert project is not None, "Argument `project` required for wandb logging"
        
        run = wandb.init(
            project=project,
            config=config, 
            tags=tags
        )
        
        for k,dataframe in results.items():
            table = wandb.Table(dataframe=dataframe)
            run.log({k: table})
        
        run.finish()
        
    def get_runs_by_filters(self, filters, entity=None, project=None):        
        api, entity, project = self.get_wandb_api(entity, project)        
        runs = api.runs(entity + "/" + project, filters=filters)
        return runs
    
    def get_latest_run_by_filters(self, filters, entity=None, project=None):
        runs = self.get_runs_by_filters(filters, entity=entity, project=project)
        N = len(runs)
        
        if N==0:
            print("No runs match the given filters")
            latest_run = None
        elif N==1:
            latest_run = runs[0]
            print(f"One run matches the given filters: {latest_run.name}, {latest_run.created_at}")
        else:
            # Sort the runs by creation date (most recent first)
            sorted_runs = sorted(runs, key=lambda run: run.created_at, reverse=True)
            latest_run = sorted_runs[0]  # This is the most recent run
            print(f"{N} runs matched the given filters, returning latest run: {latest_run.name}, {latest_run.created_at}")
            
        return latest_run
    
    def get_runs_by_id(self, hash_id, entity=None, project=None):
        assert hash_id is not None, "You must supply a hash_id (uniquely identifying the model) to use wandb"
        
        api, entity, project = self.get_wandb_api(entity, project)
        
        filters = self.wandb_filters
        filters['config.hash_id'] = hash_id
        runs = api.runs(entity + "/" + project, filters=filters)
        return runs
    
    def get_latest_run_by_id(self, hash_id, entity=None, project=None):
        runs = self.get_runs_by_id(hash_id, entity=entity, project=project)
        N = len(runs)
        
        if N==0:
            print("No runs match the given filters")
            latest_run = None
        elif N==1:
            latest_run = runs[0]
            print(f"One run matches the given filters: {latest_run.name}, {latest_run.created_at}")
        else:
            # Sort the runs by creation date (most recent first)
            sorted_runs = sorted(runs, key=lambda run: run.created_at, reverse=True)
            latest_run = sorted_runs[0]  # This is the most recent run
            print(f"{N} runs matched the given filters, returning latest run: {latest_run.name}, {latest_run.created_at}")
            
        return latest_run
        
    def fetch_results(self, **kwargs):    
        filters = {f"config.{k}":v for k,v in kwargs.items()}
        print(f"==> Search wandb with filters: {filters}")
        with suppress_all_outputs():
            api, entity, project = self.get_wandb_api(entity=None, project=None)
            filters = {**gsi.wandb_filters, **filters}
            runs = api.runs(entity + "/" + project, filters=filters)
        print(f"==> Found {len(runs)} results:")
        data = []
        for run in runs:
            results = get_all_tables(run)
            data.append((run,results))
                
        return data
    
    def fetch_run_results(self, run):
        raise NotImplementedError("Subclasses of ModelEvalWandb should implement `fetch_run_results`.")