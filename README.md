# probes
testing out some probes

# use venv
```
# make sure to load the Mamba/Conda module on the cluster
module load Mambaforge/23.3.1-fasrc01

# make sure we have cuda toolkit and GCC needed to build modules
module load cuda/12.0.1-fasrc01
module load gcc/12.2.0-fasrc01

python3 -m venv $VENV_DIR/analysis
source $VENV_DIR/analysis/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=analysis --display-name="analysis"
```