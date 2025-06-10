ROOT=${PWD} 

### create conda environment ###
conda create -y -n activegamer python=3.8 cmake=3.14.0

### activate conda environment ###
source activate activegamer

### Setup habitat-sim ###
cd ${ROOT}/third_party
git clone git@github.com:Huangying-Zhan/habitat-sim.git habitat_sim
cd habitat_sim
pip install -r requirements.txt
python setup.py install --headless --bullet

### extra installation ###
pip install opencv-python
conda install -y ipython
pip install mmcv==2.0.0
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html; \

### activegamer installation ###
pip install -r ${ROOT}/envs/requirements.txt
