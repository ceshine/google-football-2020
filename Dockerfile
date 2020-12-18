FROM nvidia/cuda:11.0-base-ubuntu20.04

LABEL maintainer="Ceshine Lee"

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instal basic utilities
RUN cp /etc/apt/sources.list /etc/apt/sources.list~ && \
    sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y   git wget sudo build-essential ca-certificates cmake libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev && \
    apt-get build-dep -y python-pygame && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# Install mamba
RUN conda install -y mamba -c conda-forge

ADD ./environment.yml .
RUN mamba env update --file ./environment.yml &&\
    conda clean -tipy


# Install the Google Football Environment
RUN git clone -b v2.8 https://github.com/google-research/football.git && \
    mkdir -p football/third_party/gfootball_engine/lib && \
    wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
# cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip install .    

RUN sed -Ei 's/action_set_v1/action_set_v2/' /home/docker/football/gfootball/env/football_action_set.py && \
    cd football && pip install .    

# For interactive shell
RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc

RUN pip install --upgrade pip && \
    pip install https://github.com/ceshine/pytorch-helper-bot/archive/google-football-2020.zip && \
    rm -rf ~/.cache/pip