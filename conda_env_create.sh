#!/bin/bash

# conda env create --prefix "$CONDA_ENV_DIR/analysis" -f environment.yml
mamba env update --file environment.yml

# Determine the Conda root directory
default_conda_envs_dir=$(conda config --show envs_dirs | grep "$USER" | awk '{print $2}')

echo "default_conda_envs_dir: $default_conda_envs_dir"

ln -s $CONDA_ENV_DIR/analysis $default_conda_envs_dir/analysis || echo "Failed to create symbolic link. Continuing..."

source activate analysis
mamba install \
    albumentations \
    attrs \
    einops \
    fastargs \
    ipykernel \
    nbconvert \
    nbformat \
    numpy \
    numba \
    fastprogress \
    tqdm \
    kornia \
    seaborn \
    torchmetrics \

# need to run "chmod u+x conda_env_create.sh" once before you can run the conda_env_create.sh script.

# example call, must specify </conda_root_dir> </prefix/path/env_name> </path/to/ffcv-ssl/repo>:
# ./conda_env_create.sh /n/holystore01/LABS/<labname>/Users/<username>/conda/ffcv-ssl envfilename
# ./conda_env_create.sh /n/holystore01/LABS/alvarez_lab/Users/alvarez/conda/ffcv-ssl envfilename
# ./conda_env_create.sh /home/jovyan/.envs/ffcv-ssl environment_ffcv.yml
# ./conda_env_create.sh $CONDA_ENV_DIR/test2 environment_test2.yml
# ./conda_env_create.sh $CONDA_ENV_DIR/workshop environment_workshop.yml

if [ -z "$1" ]; then
  echo "Error: Missing input. Usage: ./env_create.sh </path/to/environments/envname>"
  exit 1
fi

# ffcvdir=/tmp/FFCV-SSL
prefix=$1
# If envfilename is provided, use it. Otherwise, default to "environment.yml"
envfilename=${2:-environment.yml}

envname=$(basename "$prefix")
prefix_dir=$(dirname "$prefix")

# Check if the prefix directory exists
if [ ! -d "$prefix_dir" ]; then
    echo "Error: The directory $prefix_dir does not exist."
    exit 1
fi

# Check if the environment file exists
if [ ! -f "$envfilename" ]; then
    echo "Error: The environment file $envfilename does not exist."
    exit 1
fi

echo "conda env prefix: $prefix"
echo "conda envname: $envname"
echo "environment file: $envfilename"
# echo "ffc-ssl path: $ffcvdir"

# Capture the current directory
current_dir=$(pwd)

# Determine the Conda root directory
default_conda_envs_dir=$(conda config --show envs_dirs | grep "$USER" | awk '{print $2}')

echo "default_conda_envs_dir: $default_conda_envs_dir"

# make sure to load the Mamba/Conda module on the cluster
module load Mambaforge/23.3.1-fasrc01

# make sure we have cuda toolkit and GCC needed to build modules
module load cuda/11.3.1-fasrc01
module load gcc/9.5.0-fasrc01

# make sure we're in the base environment
conda deactivate

# install from environment .yml file
conda env create --prefix $prefix -f $envfilename

# create a symbolic link to the new environment
ln -s $prefix $default_conda_envs_dir/$envname || echo "Failed to create symbolic link. Continuing..."

# Activate new environment
source activate $envname

# Enforce specific pip install target location

# Determine the Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Construct the target path. No command substitution is needed here.
TARGET_PATH="$CONDA_PREFIX/lib/python$PYTHON_VERSION/site-packages"

# Create or overwrite the pip.conf file in your Conda environment's directory
echo -e "[install]\ntarget=$TARGET_PATH" > $CONDA_PREFIX/pip.conf

pip install -r requirements.txt --upgrade

# remove the pip.conf (forced target) when installing ffcv
rm $CONDA_PREFIX/pip.conf
conda deactivate
source activate $envname
pip install -e git+https://github.com/facebookresearch/FFCV-SSL.git#egg=FFCV-SSL

# ?once again add the pip.conv so future pip installs go in the right place
# echo -e "[install]\ntarget=$TARGET_PATH" > $CONDA_PREFIX/pip.conf

# install the model_rearing_workshop package itself into the environment
pip install -e . --user

# install ipykernel so you can choose 'workshop' as a kernel in jupyterlab
python -m ipykernel install --user --name=$envname