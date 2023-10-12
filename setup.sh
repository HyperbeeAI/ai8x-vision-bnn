#!/bin/bash
sudo apt-get update
if ! command -v conda &> /dev/null; then    
        curl -Lk https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh > miniconda_installer.sh
        chmod u+x miniconda_installer.sh
        bash miniconda_installer.sh -b -p ./conda -u
        conda/bin/conda init bash
        source conda/etc/profile.d/conda.sh
        conda create -y -k --prefix ./venv python=3.8.10
        conda activate ./venv/
fi
CONDA_PATH=$(which conda)
export PATH="$(dirname $CONDA_PATH)/../bin:$PATH"
source ~/.bashrc
conda create -y -k --prefix ./venv python=3.8.10
source activate ./venv/

pip install -r requirements.txt --no-cache-dir
echo "Setup completed."
