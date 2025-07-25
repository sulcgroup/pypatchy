# use latest ubuntu version?
FROM ubuntu:22.04
# “don’t prompt me for any questions”
ENV DEBIAN_FRONTEND=noninteractive

# update ubuntu version???
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get clean

# set timezone to phoenix???
RUN ln -snf /usr/share/zoneinfo/America/Phoenix /etc/localtime && \
    echo "America/Phoenix" > /etc/timezone && \
    echo 'export PS1="\[\e[1;32m\]\u@\h:\[\e[1;34m\]\w\[\e[0m\]$ "' >> /root/.bashrc && \
    echo 'alias ll="ls -alF"' >> /root/.bashrc && \
    echo 'alias la="ls -A"' >> /root/.bashrc && \
    echo 'alias l="ls -CF"' >> /root/.bashrc && \
    echo 'alias h="history"' >> /root/.bashrc && \
    echo 'export CLICOLOR=1' >> /root/.bashrc && \
    echo 'export LSCOLORS=GxFxCxDxBxegedabagaced' >> /root/.bashrc && \
    echo "shopt -s autocd">> /root/.bashrc; echo 'export PATH=/opt/venv/bin:/root/Github/oxDNA/build/bin:$PATH' >> /root/.bashrc
    #echo 'source /etc/bash_completion' >> /root/.bashrc &&\

# we will have a /root/Github
# install stuff that is needed
RUN apt-get install -y build-essential cmake git wget curl neovim python3 python3-pip python3-venv g++ gcc libeigen3-dev libboost-dev libboost-regex-dev libspdlog-dev libboost-graph-dev pybind11-dev python3-dev openssh-client && \
    apt-get clean && \
    mkdir -p /root/Github
# copy current directory to root/Github/pypatchy
COPY . /root/Github/pypatchy

# use the polycubes cpp build directory for the next few

# use bash -l for subsequent RUNs so ~/.bashrc is loaded
SHELL ["/bin/bash","-lc"]

# install micromamba
RUN curl -L micro.mamba.pm/install.sh | bash

# clone oxDNA
#RUN --mount=type=ssh git clone git@github.com:lorenzo-rovigatti/oxDNA.git /root/Github/oxDNA
# todo: replace with the above line of code once i get the flavio code PR'd into Lorenzo's
# get back to this
RUN --mount=type=ssh git clone git@github.com:JeDeveloper/oxDNA.git /root/Github/oxDNA

# clone the ipy_oxdna repo
# this will use whatever keys your local ssh-agent has loaded
# todo: stabilize the stack and make this point to matt's ipy_oxDNA version
RUN --mount=type=ssh git clone git@github.com:JeDeveloper/ipy_oxDNA.git /root/Github/ipy_oxDNA

# clone polycubes git repo
RUN --mount=type=ssh git clone git@github.com:sulcgroup/polycubes-clone.git /root/github/polycubes

# create & activate the pypatchy env, then pip‑install both packages
# todo: check if we really need to install with -e in the docker version
RUN micromamba create -n pypatchy -f /root/Github/pypatchy/pypatchy.yml --channel-priority flexible \
 && micromamba activate pypatchy \
 && pip install -e /root/Github/ipy_oxDNA \
 && pip install -e /root/Github/pypatchy

# purge pip cache
pip cache purge

WORKDIR /root/Github/pypatchy
# set up pypatchy stuff
RUN python setup.py

# make pypatchy run directory
# todo make this settable or something for systems with seperate home and scratch drives
RUN mkdir /root/patchy_run
# setup.py creates the .pypatchy directory
# replace the default settings.cfg file with a custom one for docker setup
run echo "[ANALYSIS] \n \
    simulation_data_dir = /root/patchy_run \
    tlm_data_dir = /home/josh/lab_data/tlm_data/raws \
    [SETUP] \
    server_config = docker" > /root/.pypatchy/settings.cfg

# create server configs
# todo: make these settable outside dockerfile?
# todo: option to set up without cuda
# todo: sed the rovigatti .h file that has the max patches
RUN echo '{ \
	"input_file_params": { \
		"backend": "CUDA",\
        "backend_precision": "float", \
		"interaction_type": "DetailedPatchySwapInteraction", \
        "plugin_search_path": "/root/Github/oxDNA/contrib/rovigatti", \
        "DPS_interaction_matrix_file": "interactions.txt", \
		"max_density_multiplier": 10, \
		"DPS_normalise_patches": 0, \
		"DPS_sigma_ss": 0.082 \
	}, \
	"oxdna_path": "/root/Github/oxDNA/", \
	"is_slurm": false, \
	"patchy_format": "lorenzo", \
	"cuda_mps": false \
}' > /root/.pypatchy/spec_files/server_configs/lr.json

RUN echo '{ \
	"input_file_params": { \
		"backend": "CPU", \
        "backend_precision": "mixed", \
		"interaction_type": "PatchyShapeInteraction", \
        "plugin_search_path": "/root/Github/oxDNA/contrib/romano", \
        "patchy_file": "patches.txt", \
        "particle_file": "particles.txt", \
        "interaction_potentials": [ \
        	"fr_excl_vol", \
        	"fr_patchy" \
        ], \
      "PATCHY_alpha": 0.082 \
    }, \
    "no_oxpy_check": true, \
	"oxdna_path": "/root/Github/oxDNA/", \
	"is_slurm": false, \
	"patchy_format": "flavio", \
	"cuda_mps": false \
' > /root/.pypatchy/spec_files/server_configs/fr.json

# compile oxDNA, with python bindings associated with pypatchy
WORKDIR /root/Github/oxDNA
RUN mkdir build
# todo: option to install without cuda
WORKDIR build
# cmake oxDNA
RUN cmake .. -DCUDA=ON -DPYTHON=ON
# compile oxDNA + python bindings
RUN make install -j$(nproc)
# compile flavio romano patchy model
RUN make romano -j$(nproc)
# compile lorenzo rovigatti patchy model
RUN make rovigatti -j$(nproc)
# TODO: compile Josh Evans patchy model


# cmake and compile  polycubes library
WORKDIR /root/Github/polycubes/cpp
RUN mkdir build
WORKDIR build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYBIND11_PYTHON_VERSION=3.9 \
    -DPYTHON_EXECUTABLE=$(which python) \
 && make polycubes_py -j$(nproc)

RUN cp src/libtlm.so $(python -m side --user-site)

# run polycubes tests
# run ipy_oxDNA tests
# run pypatchy tests

# start bash
ENTRYPOINT ["bash", "-l"]