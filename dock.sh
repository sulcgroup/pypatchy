#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# SSH agent setup: the `docker build` below forwards this agent into the build
# (--ssh default) so the `git clone git@github.com:...` steps in the Dockerfile
# can authenticate.
# ---------------------------------------------------------------------------
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ ! -f "$SSH_KEY" ]]; then
    echo "ERROR: ssh key not found at $SSH_KEY (override with SSH_KEY=/path/to/key)" >&2
    exit 1
fi

eval "$(ssh-agent -s)"
# kill the agent we just started when the script exits, however it exits
trap 'ssh-agent -k >/dev/null 2>&1 || true' EXIT

if ! ssh-add "$SSH_KEY"; then
    echo "ERROR: failed to add $SSH_KEY to the ssh-agent" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Preflight: verify we can actually reach every git repo the Dockerfile clones,
# on the specific branch it checks out, BEFORE kicking off a long build that
# would otherwise fail partway through with a less obvious error.
#
# Entries are "<url>" or "<url>#<branch>". Keep in sync with the `git clone`
# lines in the Dockerfile.
# ---------------------------------------------------------------------------
REQUIRED_REPOS=(
    "git@github.com:lorenzo-rovigatti/oxDNA.git"
    "git@github.com:sulcgroup/pypatchy.git#unstable"
    "git@github.com:sulcgroup/oxpy-utils.git"
    "git@github.com:sulcgroup/polycubes-clone.git#nightmare"
    "https://github.com/lorenzo-rovigatti/tacoxDNA.git"
)

# non-interactive git/ssh: never prompt for a password, passphrase, or unknown
# host key -- just fail so the preflight can report it.
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

echo "Checking access to required git repositories..."
repo_check_failed=0
for spec in "${REQUIRED_REPOS[@]}"; do
    url="${spec%%#*}"
    branch=""
    [[ "$spec" == *"#"* ]] && branch="${spec#*#}"

    if [[ -n "$branch" ]]; then
        if ! out="$(git ls-remote --heads "$url" "$branch" 2>&1)"; then
            echo "  MISSING  $url  --  cannot access repo:"
            echo "$out" | sed 's/^/               /'
            repo_check_failed=1
        elif [[ -z "$out" ]]; then
            echo "  MISSING  $url  --  branch '$branch' not found"
            repo_check_failed=1
        else
            echo "  ok       $url  (branch: $branch)"
        fi
    else
        if ! out="$(git ls-remote "$url" HEAD 2>&1)"; then
            echo "  MISSING  $url  --  cannot access repo:"
            echo "$out" | sed 's/^/               /'
            repo_check_failed=1
        else
            echo "  ok       $url"
        fi
    fi
done

if [[ "$repo_check_failed" -ne 0 ]]; then
    echo >&2
    echo "ERROR: one or more required repositories are not accessible with the" >&2
    echo "current ssh key ($SSH_KEY). Fix access (request repo permissions, or" >&2
    echo "add the right key) before rebuilding the image." >&2
    exit 1
fi
echo "All required repositories are accessible."
echo

DOCKER_BUILDKIT=1 docker build -t patchy:latest --ssh default -f Dockerfile --no-cache .;

# run polycubes tests
docker run --rm patchy:latest \
    -lc "cd /root/Github/polycubes/cpp/build && ctest --output-on-failure -j$(nproc)"

# run oxpy-utils tests
# test_seqdesign.py is excluded: it imports the `nupack` package, which is
# licensed software not installable via pip/conda and isn't set up in this
# image, so collecting it always fails with ModuleNotFoundError.
docker run --rm patchy:latest \
  -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/oxpy-utils && pytest --maxfail=1 -q --ignore=tests/test_seqdesign.py"

# run pypatchy tests
docker run --rm patchy:latest \
  -l -c "\
    eval \"\$(micromamba shell hook --shell bash)\" && \
    micromamba activate pypatchy && \
    cd /root/Github/pypatchy && pytest --maxfail=1 -q"

docker tag patchy:latest joshuaevanslowell/oxdna-pypatchy:latest;
docker push joshuaevanslowell/oxdna-pypatchy:latest;
