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
    libeigen3-dev libboost-dev libboost-regex-dev libspdlog-dev libboost-graph-dev pybind11-dev python3-dev gfortran libopenblas-dev \
    openssh-client && \
    apt-get clean && \
    mkdir -p /root/Github
# copy current directory to root/Github/pypatchy
#COPY . /root/Github/pypatchy

# container-level CLAUDE.md so Claude Code has cross-repo context regardless of which repo it's launched in
COPY docker/ROOT_CLAUDE.md /root/CLAUDE.md

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
RUN --mount=type=ssh git clone git@github.com:lorenzo-rovigatti/oxDNA.git /root/Github/oxDNA
RUN --mount=type=ssh git clone git@github.com:sulcgroup/pypatchy.git /root/Github/pypatchy && cd /root/Github/pypatchy && git checkout unstable
RUN --mount=type=ssh git clone git@github.com:sulcgroup/oxpy-utils.git /root/Github/oxpy-utils
RUN --mount=type=ssh git clone git@github.com:sulcgroup/polycubes-clone.git /root/Github/polycubes \
    && cd /root/Github/polycubes \
    && git checkout nightmare && \
# create & activate the pypatchy env, then pip‑install both packages
# todo: check if we really need to install with -e in the docker version
micromamba create -y -n pypatchy -f /root/Github/pypatchy/pypatchy.yml && \
micromamba run -n pypatchy micromamba install -y boost-cpp --channel conda-forge \
 && micromamba clean --all --yes

# generate a VS Code multi-root workspace file covering ~/scripts and every ~/Github/<repo>,
# so attaching VS Code (Dev Containers / Remote-SSH) to this image opens all the cloned repos at once
RUN mkdir -p /root/scripts
RUN python3 <<'PYEOF'
import json, os

root = "/root"
folders = [{"path": "scripts"}]
gh = os.path.join(root, "Github")
for name in sorted(os.listdir(gh)):
    folders.append({"path": f"Github/{name}"})

with open(os.path.join(root, "pypatchy.code-workspace"), "w") as f:
    json.dump({"folders": folders, "settings": {}}, f, indent=2)
PYEOF

# update docker env so we can find micromamba stuff later
# 1. Tell Docker where your micromamba root actually lives
ENV MAMBA_ROOT_PREFIX=/root/.local/share/mamba

# 2. Prepend the pypatchy env’s bin dir to PATH
ENV PATH=$MAMBA_ROOT_PREFIX/envs/pypatchy/bin:$PATH

RUN micromamba run -n pypatchy pip install -e /root/Github/oxpy-utils \
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
mkdir -p /root/data/tlm_data && \
echo -e "[ANALYSIS] \n \
    simulation_data_dir = /root/patchy_run \n \
    tlm_data_dir = /root/data/tlm_data \n \
    [SETUP] \n \
    server_config = docker" > /root/.pypatchy/settings.cfg && \
echo -e '{ \n \
	"input_file_params": { \n \
		"backend": "CPU", \n \
        "backend_precision": "mixed", \n \
        "interaction_potentials": [ \n \
        	"fr_excl_vol",  \n \
        	"fr_patchy" \n \
        ], \n \
      "PATCHY_alpha": 0.082 \n \
    }, \n \
    "no_oxpy_check": true, \n \
	"oxdna_path": "/root/Github/oxDNA/", \n \
	"is_slurm": false, \n \
	"patchy_format": "flavio", \n \
	"cuda_mps": false \n \
' > /root/.pypatchy/spec_files/server_configs/fr.json && \
cp /root/.pypatchy/spec_files/server_configs/fr.json /root/.pypatchy/spec_files/server_configs/docker.json && \
cp /root/Github/pypatchy/spec_files/input_files/default.json /root/.pypatchy/spec_files/input_files/ && \
cp /root/Github/pypatchy/spec_files/observables/plclustertopology.json /root/.pypatchy/spec_files/observables/ && \
cp /root/Github/pypatchy/spec_files/observables/plclustertopology.json /root/.pypatchy/spec_files/observables/ &&\
echo -e '{ \n \
	"name": "clusters.txt", \n \
	"print_every": 1e7, \n \
	"cols": [ \n \
		{ \n \
			"name": "PatchyBonds", \n \
			"show_types": 1 \n \
		} \n \
	] \n \
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
    make polycube_tests -j$(nproc)
# note: libtlm.so/.pyi are installed by the POST_BUILD custom command in
# polycubes/cpp/src/CMakeLists.txt, which copies to the active interpreter's
# sysconfig purelib (the pypatchy env's real site-packages). Do NOT also `cp`
# to `python -m site --user-site` here: that path is always ~/.local/lib/...,
# independent of which env is active, and it sits ahead of the env's
# site-packages on sys.path -- so a copy there silently shadows the correct
# one and goes stale on every rebuild that doesn't re-run this Dockerfile step.


# putting this at the end because initially i forgot to do it at all, and
# running it later didn't blow anything up
RUN pip install python-sat

RUN cd ~/Github/oxDNA/analysis && bash build.sh

# start bash
#ENTRYPOINT ["/bin/bash", "-c"]
#CMD ["if [ $# -eq 0 ]; then echo \"Starting pypatchy\" && eval \"$(micromamba shell hook --shell bash)\" && /bin/bash; else exec \"$@\"; fi"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo 'micromamba activate pypatchy' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD ["--login"]

# One and build and upload with
# docker build -t pypatchy:latest .
# docker run -it --rm --gpus all -v "$(pwd):/mnt" subhajitroy/private:pypatchy # To start the container with GPU support and will remove the container after exit or when the command finishes
