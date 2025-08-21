# syntax=docker/dockerfile:1.4

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
RUN apt-get install -y build-essential cmake git wget curl neovim python3 python3-pip python3-venv g++ gcc \
    libeigen3-dev libboost-dev libboost-regex-dev libspdlog-dev libboost-graph-dev pybind11-dev python3-dev \
    openssh-client && \
    apt-get clean && \
    mkdir -p /root/Github
# copy current directory to root/Github/pypatchy
COPY . /root/Github/pypatchy

# use the polycubes cpp build directory for the next few
# make sure SSH can verify GitHub’s host key
RUN mkdir -p /root/.ssh \
 && chmod 700 /root/.ssh \
 && ssh-keyscan github.com >> /root/.ssh/known_hosts

# use bash -l for subsequent RUNs so ~/.bashrc is loaded
SHELL ["/bin/bash","-lc"]

# install micromamba

# STEP: install micromamba static binary into /usr/local/bin
RUN apt-get update && apt-get install -y bzip2 ca-certificates \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xjC /usr/local/bin --strip-components=1 bin/micromamba \
 && chmod +x /usr/local/bin/micromamba \
 && rm -rf /var/lib/apt/lists/* \
 && micromamba --version
# clone oxDNA
#RUN --mount=type=ssh git clone git@github.com:lorenzo-rovigatti/oxDNA.git /root/Github/oxDNA
# todo: replace with the above line of code once i get the flavio code PR'd into Lorenzo's
# get back to this
RUN --mount=type=ssh git clone git@github.com:JeDeveloper/oxDNA.git /root/Github/oxDNA && \
git clone git@github.com:JeDeveloper/ipy_oxDNA.git /root/Github/ipy_oxDNA && \
git clone git@github.com:sulcgroup/polycubes-clone.git /root/Github/polycubes \
    && cd /root/Github/polycubes \
    && git checkout nightmare && \
# create & activate the pypatchy env, then pip‑install both packages
# todo: check if we really need to install with -e in the docker version
micromamba create -y -n pypatchy -f /root/Github/pypatchy/pypatchy.yml --channel-priority flexible && \
micromamba run -n pypatchy micromamba install -y boost-cpp --channel conda-forge \
 && micromamba clean --all --yes

# update docker env so we can find micromamba stuff later
# 1. Tell Docker where your micromamba root actually lives
ENV MAMBA_ROOT_PREFIX=/root/.local/share/mamba

# 2. Prepend the pypatchy env’s bin dir to PATH
ENV PATH=$MAMBA_ROOT_PREFIX/envs/pypatchy/bin:$PATH

RUN micromamba run -n pypatchy pip install -e /root/Github/ipy_oxDNA \
    && micromamba run -n pypatchy pip install -e /root/Github/pypatchy \
    && micromamba clean --all --yes \
    && micromamba run -n pypatchy pip cache purge \
    && mkdir /root/.pypatchy  \
    && mkdir /root/.pypatchy/spec_files \
    && mkdir /root/.pypatchy/spec_files/server_configs \
    && mkdir /root/.pypatchy/spec_files/input_files \
    && mkdir /root/.pypatchy/spec_files/observables \
    && mkdir /root/.pypatchy/input  \
    && mkdir /root/.pypatchy/input/topologies \
    && mkdir /root/.pypatchy/input/targets \
    && mkdir /root/.pypatchy/input/tlm \
    && mkdir /root/.pypatchy/output  \
    && mkdir /root/.pypatchy/output/SAT \
    && mkdir /root/.pypatchy/output/logs \
    && mkdir /root/.pypatchy/output/logs/SAT


# make pypatchy run directory
# todo make this settable or something for systems with seperate home and scratch drives
RUN mkdir /root/patchy_run && \
echo "[ANALYSIS] \n \
    simulation_data_dir = /root/patchy_run \
    tlm_data_dir = /home/josh/lab_data/tlm_data/raws \
    [SETUP] \
    server_config = docker" > /root/.pypatchy/settings.cfg && \
echo '{ \
	"input_file_params": { \
		"backend": "CPU", \
        "backend_precision": "mixed", \
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
' > /root/.pypatchy/spec_files/server_configs/fr.json && \
cp /root/Github/pypatchy/spec_files/input_files/default.json /root/.pypatchy/spec_files/input_files/ && \
cp /root/Github/pypatchy/spec_files/observables/plclustertopology.json /root/.pypatchy/spec_files/observables/ && \
cp /root/Github/pypatchy/spec_files/observables/plclustertopology.json /root/.pypatchy/spec_files/observables/ &&\
echo '{ \
	"name": "clusters.txt", \
	"print_every": 1e7, \
	"cols": [ \
		{ \
			"name": "PatchyBonds", \
			"show_types": 1 \
		} \
	] \
}' > /root/Github/pypatchy/spec_files/observables/patchbonds.json

# compile oxDNA, with python bindings associated with pypatchy
WORKDIR /root/Github/oxDNA
RUN mkdir build
# todo: option to install without cuda
WORKDIR build
# cmake oxDNA
# TODO: add -DCUDA=ON and make docker set up cuda
RUN cmake .. -DPYTHON=ON && \
# compile oxDNA + python bindings
make install -j$(nproc) && \
# compile flavio romano patchy model
make romano -j$(nproc) && \
# compile lorenzo rovigatti patchy model
make rovigatti -j$(nproc) && \
# TODO: compile Josh Evans patchy model
# cmake and compile  polycubes library
cd /root/Github/polycubes/cpp && \
mkdir build && cd build && \
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYBIND11_PYTHON_VERSION=3.9 \
    -DPYTHON_EXECUTABLE=$(which python) \
 && make polycubes_py -j$(nproc) && \
cp src/libtlm.so $(python -m site --user-site)




# start bash
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["if [ $# -eq 0 ]; then echo \"Starting pYpatchy\" && /bin/bash; else exec \"$@\"; fi"]

# One and build and upload with 
# docker build -t pypatchy:latest .
# docker run -it --rm --gpus all -v "$(pwd):/mnt" subhajitroy/private:pypatchy # To start the container with GPU support and will remove the container after exit or when the command finishes