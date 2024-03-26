import pandas as pd
import json
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
        
__all__ = ['list_artifacts', 'get_table_by_name', 'get_all_tables', 'suppress_all_outputs']

def list_artifacts(run):
    for artifact in run.logged_artifacts():
        print(artifact.name, artifact.type)
        
def get_table_by_name(run, table_name):
    for artifact in run.logged_artifacts():
        # get the name of these entries
        entries = [entry.split(".table")[0] for entry in artifact.manifest.entries]
        if table_name not in entries: continue            
        
        # looks like we have our entry
        artifact_dir = artifact.download()  # Downloads all files in the artifact
        entry = artifact.get_entry(table_name)
        json_file_path = f"{artifact_dir}/{entry.path}"

        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data=data['data'], columns=data['columns'])
        
        return df

def get_all_tables(run):
    results = dict()
    for artifact in run.logged_artifacts():
        entries = [entry.split(".table")[0] for entry in artifact.manifest.entries if '.table' in entry]
        for table_name in entries:
            with suppress_all_outputs():
                artifact_dir = artifact.download()  # Downloads all files in the artifact
            entry = artifact.get_entry(table_name)
            json_file_path = f"{artifact_dir}/{entry.path}"

            with open(json_file_path, 'r') as file:
                data = json.load(file)

            df = pd.DataFrame(data=data['data'], columns=data['columns'])
            results[table_name] = df
        
    return results
    
@contextmanager
def suppress_all_outputs():
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr    