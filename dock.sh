#!/bin/bash
set -e
eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/id_ed25519
DOCKER_BUILDKIT=1 docker build -t patchy:latest --ssh default -f Dockerfile --no-cache .;

# run polycubes tests
docker run --rm patchy:latest \
    bash -lc "cd /root/Github/polycubes/cpp/build && ctest --output-on-failure -j$(nproc)"

# run oxpy-utils tests
docker run --rm patchy:latest \
  bash -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/oxpy-utils && pytest --maxfail=1 -q"

# run pypatchy tests
docker run --rm patchy:latest \
  bash -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/pypatchy && pytest --maxfail=1 -q"

docker tag patchy:latest joshuaevanslowell/oxdna-pypatchy:latest;
docker push joshuaevanslowell/oxdna-pypatchy:latest;
