docker build -t patchy:latest .
# run polycubes tests
docker run --rm patchy:latest \
    bash -lc "cd /root/Github/polycubes/cpp/build && ctest --output-on-failure -j$(nproc)"

# run ipy_oxDNA tests
docker run --rm patchy:latest \
  bash -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/ipy_oxDNA && pytest --maxfail=1 -q"

# run pypatchy tests
docker run --rm patchy:latest \
  bash -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/pypatchy && pytest --maxfail=1 -q"


