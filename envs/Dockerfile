##################################################
# Dockerfile for ActiveGAMER
# Ver.1.0 (20250521; Huangying Zhan)
#   First release
##################################################
### this Dockerfile is adapted from: ###
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

##################################################
### ENV and ARG variables
##################################################
ARG http_proxy=""
ARG https_proxy=""
ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy=localhost,127.0.0.1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.8

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

##################################################
### System package installation
##################################################
### Temporary: due to GPG error "public key is not available" in Ubuntu 20.04 CUDA 11.4.0 ###
### > see: https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716 ###
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    git \
    build-essential \
    graphviz \
    ffmpeg \
    cmake curl \
    libpng-dev \
    libjpeg-dev libjpeg-dev \
    libpng++-dev \
    libgl1-mesa-dev \
    ninja-build libglib2.0-0 libglib2.0-dev libsm6 libxrender-dev libxext6 \
    vim tmux \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    libgtk2.0-0 libcanberra-gtk-module \
    python${PYTHON_VERSION}-tk \
    libboost-all-dev libeigen3-dev libpcl-dev \
    libopenexr-dev \
    screen \
    wget \
    zip \
    unzip \
    # for Pangolin
    libglew-dev \
    # habitat
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    && apt-get clean 

##################################################
### Environment variable setup
##################################################
### PATH ###
ENV PATH=/home/$USER_NAME/.local/bin:$PATH
ENV PATH=/home/$USER_NAME/bin:$PATH

### color prompt ###
ENV TERM='xterm-color'

##################################################
### Python packages installation
##################################################
### pip ###
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

##################################################
### Anaconda (miniconda)
##################################################
### install miniconda ###
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

### Run conda init ###
RUN /opt/conda/bin/conda init

### create conda environment ###
RUN conda create -n activegamer python=3.8 cmake=3.14.0

### install packages in conda environment ###
RUN mkdir /tmp/envs
COPY envs/* /tmp/envs/
RUN git clone https://github.com/Huangying-Zhan/habitat-sim.git

### install habitat-sim ###
RUN /bin/bash -c ". activate activegamer; \
                    cd habitat-sim; \
                    pip install -r requirements.txt; \
                    python setup.py install --headless --bullet;" 

### install other packages ###
RUN /bin/bash -c ". activate activegamer; \
                    conda install ipython; \
                    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html; \
                    pip install -r /tmp/envs/requirements.txt"

##################################################
### Clean tmp files
##################################################
RUN rm -rf /var/lib/apt/lists/* 
RUN rm -rf /tmp/* 
RUN rm -rf /var/tmp/*

##################################################
### User creation (non-root)
##################################################
### Create a non-root user and switch to it. These args can be updated externally ###
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME='dummy'
ARG GROUP_NAME='dummy'

### Create a group ###
RUN addgroup --gid $GROUP_ID $GROUP_NAME
### Create a user ###
RUN useradd -rm -d /home/$USER_NAME --shell /bin/bash  --uid $USER_ID --gid $GROUP_ID -G $GROUP_NAME $USER_NAME 

EXPOSE 22
USER $USER_NAME:$GROUP_NAME
WORKDIR /home/$USER_NAME

##################################################
### Personalized System initialization commands/installations (Optional)
##################################################
### fzf ###
RUN git clone --depth 1 https://github.com/junegunn/fzf.git /tmp/.fzf
RUN /tmp/.fzf/install --all
