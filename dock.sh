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

# ---------------------------------------------------------------------------
# Build, test, and (optionally) publish. Each step runs as a named "stage":
# a stage failure is recorded and, unless KEEP_GOING=1, later stages are
# skipped. Publishing only happens if every earlier stage passed. A summary
# is printed at the end and the script exits non-zero if anything failed.
#
# Env toggles:
#   SKIP_TESTS=1   build (and publish, unless SKIP_PUSH) but run no tests
#   SKIP_PUSH=1    build and test but don't `docker tag` / `docker push`
#   KEEP_GOING=1   run every stage even after one fails (still exits non-zero)
# ---------------------------------------------------------------------------
IMAGE_LOCAL="patchy:latest"
IMAGE_REMOTE="joshuaevanslowell/oxdna-pypatchy:latest"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
KEEP_GOING="${KEEP_GOING:-0}"

export DOCKER_BUILDKIT=1

stage_names=()
stage_results=()
any_failed=0

# run_stage "<name>" cmd arg...   -- run cmd, record pass/fail, never abort here
run_stage() {
    local name="$1"; shift
    if [[ "$any_failed" -ne 0 && "$KEEP_GOING" -ne 1 ]]; then
        echo ">>> SKIP  $name  (an earlier stage failed; set KEEP_GOING=1 to run anyway)"
        stage_names+=("$name"); stage_results+=("skip")
        return 0
    fi
    echo
    echo ">>> STAGE $name"
    local rc=0
    set +e
    ( set -x; "$@" )
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
        echo ">>> PASS  $name"
        stage_names+=("$name"); stage_results+=("pass")
    else
        echo ">>> FAIL  $name  (exit $rc)" >&2
        stage_names+=("$name"); stage_results+=("FAIL($rc)")
        any_failed=1
    fi
    return 0
}

# pytest inside the built image, in the pypatchy env. $1 = repo under
# /root/Github, remaining args = extra pytest args.
container_pytest() {
    local repo="$1"; shift
    docker run --rm "$IMAGE_LOCAL" -l -c "
        eval \"\$(micromamba shell hook --shell bash)\" &&
        micromamba activate pypatchy &&
        cd /root/Github/$repo &&
        pytest -q -rfE $*"
}

run_stage "docker build" \
    docker build -t "$IMAGE_LOCAL" --ssh default -f Dockerfile --no-cache .

if [[ "$SKIP_TESTS" -ne 1 ]]; then
    run_stage "polycubes ctest" \
        docker run --rm "$IMAGE_LOCAL" -lc \
        'cd /root/Github/polycubes/cpp/build && ctest --output-on-failure -j"$(nproc)"'

    # test_seqdesign.py is excluded: it imports the `nupack` package, which is
    # licensed software not installable via pip/conda and isn't set up in this
    # image, so collecting it always fails with ModuleNotFoundError.
    run_stage "oxpy-utils pytest" \
        container_pytest oxpy-utils --ignore=tests/test_seqdesign.py

    run_stage "pypatchy pytest" \
        container_pytest pypatchy
else
    echo ">>> SKIP  tests  (SKIP_TESTS=1)"
fi

if [[ "$SKIP_PUSH" -ne 1 ]]; then
    if [[ "$any_failed" -eq 0 ]]; then
        run_stage "docker tag"  docker tag "$IMAGE_LOCAL" "$IMAGE_REMOTE"
        run_stage "docker push" docker push "$IMAGE_REMOTE"
    else
        echo ">>> SKIP  publish  (a prior stage failed)"
    fi
else
    echo ">>> SKIP  publish  (SKIP_PUSH=1)"
fi

echo
echo "==================== SUMMARY ===================="
if [[ "${#stage_names[@]}" -gt 0 ]]; then
    for i in "${!stage_names[@]}"; do
        printf '  %-9s %s\n' "${stage_results[$i]}" "${stage_names[$i]}"
    done
fi
echo "================================================"
if [[ "$any_failed" -ne 0 ]]; then
    echo "One or more stages failed." >&2
    exit 1
fi
echo "All stages passed."
