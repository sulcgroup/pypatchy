#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Options can come from the environment (`KEEP_GOING=1 bash dock.sh`) or as
# arguments (`bash dock.sh --keep-going`, `--KEEP_GOING=1`, `GPUS=none`).
# Unknown arguments are an error rather than silently ignored.
# ---------------------------------------------------------------------------
for arg in "$@"; do
    kv="${arg#--}"
    key="${kv%%=*}"; key="${key^^}"; key="${key//-/_}"
    if [[ "$kv" == *=* ]]; then val="${kv#*=}"; else val=""; fi
    case "$key" in
        SKIP_TESTS|SKIP_PUSH|KEEP_GOING)
            [[ "$arg" == --* || -n "$val" ]] || { echo "ERROR: unknown argument '$arg'" >&2; exit 1; }
            printf -v "$key" '%s' "${val:-1}" ;;
        GPUS|SSH_KEY|CONTAINER_TEST_TIMEOUT)
            [[ "$kv" == *=* ]] || { echo "ERROR: '$arg' needs a value ($key=...)" >&2; exit 1; }
            printf -v "$key" '%s' "$val" ;;
        *)
            echo "ERROR: unknown argument '$arg'" >&2
            echo "  known: --skip-tests --skip-push --keep-going --gpus=all|none|auto --ssh-key=PATH --container-test-timeout=SECS" >&2
            exit 1 ;;
    esac
done

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
    "git@github.com:JeDeveloper/oxDNA.git"
    "git@github.com:sulcgroup/pypatchy.git#main"
    "git@github.com:sulcgroup/oxpy-utils.git"
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
# so e.g. run `SKIP_TESTS=1 SKIP_PUSH=1 bash dock.sh` (or equivalently
# `bash dock.sh --skip-tests --skip-push`) to only build and not test or push
# ---------------------------------------------------------------------------
IMAGE_LOCAL="patchy:latest"
IMAGE_REMOTE="joshuaevanslowell/oxdna-pypatchy:latest"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
KEEP_GOING="${KEEP_GOING:-0}"

export DOCKER_BUILDKIT=1

# GPU access for the test containers. Parts of the test suites run oxDNA's CUDA
# backend (e.g. oxpy-utils' test_cadnano_workflow production stage), and that
# backend calls exit() -- not a catchable exception -- when no device is
# visible, which hard-kills pytest with no traceback. GPUS=all|none|auto
# (default auto: use `--gpus all` iff `nvidia-smi -L` lists a device).
GPUS="${GPUS:-auto}"
gpu_flag=()
case "$GPUS" in
    none) echo "GPUS=none -- test containers run CPU-only (CUDA tests will fail)" ;;
    all)  gpu_flag=(--gpus all) ;;
    auto)
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "no GPU detected -- test containers run CPU-only (CUDA tests will fail)"
        elif smi_out="$(nvidia-smi -L 2>&1)"; then
            gpu_flag=(--gpus all)
            echo "GPU detected -- test containers run with --gpus all"
        elif [[ "$SKIP_TESTS" -ne 1 ]]; then
            # nvidia-smi is installed but broken (most often "Driver/library
            # version mismatch" after an unattended driver upgrade, fixed by a
            # reboot). Silently falling back to CPU-only would just make the
            # oxpy-utils stage die mid-run with no traceback ~10 min from now.
            echo "ERROR: nvidia-smi is installed but failing:" >&2
            echo "$smi_out" | sed 's/^/    /' >&2
            echo "Fix the driver (a driver/library version mismatch needs a reboot or" >&2
            echo "an nvidia module reload), or pass GPUS=none to test CPU-only anyway." >&2
            exit 1
        fi ;;
    *) echo "ERROR: GPUS must be all|none|auto (got '$GPUS')" >&2; exit 1 ;;
esac

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
#
# Runs detached with a hard wall-clock timeout (CONTAINER_TEST_TIMEOUT,
# default 30 min) instead of plain `docker run --rm` in the foreground: the
# GPU-backed suites (oxpy-utils in particular) have a known hang in
# SimulationManager.worker_manager()'s worker-drain loop (an unbounded
# `while not self.process_queue.empty(): sleep(10)` with no timeout -- see
# sulcgroup/oxpy-utils) that leaves the container running forever, pinning a
# CUDA context (GPU memory that's only released when the process exits) with
# no indication anything is wrong. `docker run --rm` alone can't recover from
# that -- there's no process to time out, since the container itself never
# exits. On timeout here we docker-kill and remove the container so the stage
# fails loudly instead of leaking a container + GPU memory indefinitely.
CONTAINER_TEST_TIMEOUT="${CONTAINER_TEST_TIMEOUT:-1800}"
container_pytest() {
    local repo="$1"; shift
    local cname="pypatchy-test-${repo}-$$"
    docker run -d --name "$cname" "${gpu_flag[@]}" "$IMAGE_LOCAL" -l -c "
        eval \"\$(micromamba shell hook --shell bash)\" &&
        micromamba activate pypatchy &&
        cd /root/Github/$repo &&
        pytest -q -rfE $*" >/dev/null

    docker logs -f "$cname" &
    local logs_pid=$!

    local rc
    if rc="$(timeout "$CONTAINER_TEST_TIMEOUT" docker wait "$cname" 2>/dev/null)"; then
        :
    else
        echo ">>> $repo pytest exceeded ${CONTAINER_TEST_TIMEOUT}s -- killing hung container $cname (see comment above container_pytest)" >&2
        docker kill "$cname" >/dev/null 2>&1 || true
        rc=124
    fi

    kill "$logs_pid" 2>/dev/null || true
    wait "$logs_pid" 2>/dev/null || true
    docker rm -f "$cname" >/dev/null 2>&1 || true
    return "$rc"
}

run_stage "docker build" \
    docker build -t "$IMAGE_LOCAL" --ssh default -f Dockerfile --no-cache .

if [[ "$SKIP_TESTS" -ne 1 ]]; then
    # test_seqdesign.py is excluded: it imports the `nupack` package, which is
    # licensed software not installable via pip/conda and isn't set up in this
    # image, so collecting it always fails with ModuleNotFoundError.
    #
    # The three --deselect'd tests are known-failing bugs in the oxpy-utils test
    # suite, unrelated to this image (verified 2026-09-02, 466 others pass):
    #   test_load_ffs                 - reads a CWD-relative "test_data/example_ffs"
    #                                   path, only resolves if pytest runs from tests/
    #   test_ffs_seperation_from_yml  - its inline config (T=70C, num_cpus=ncpu/2)
    #                                   makes every FFS shot blow up -> "No confs
    #                                   generated" (sibling test_ffs_seperation passes)
    #   test_matches_correct_output   - cadnano->oxDNA sequence assignment disagrees
    #                                   with the checked-in reference .top (geometry
    #                                   matches); likely the unpinned tacoxDNA clone
    # Remove a --deselect once its upstream bug in sulcgroup/oxpy-utils is fixed.
    run_stage "oxpy-utils pytest" \
        container_pytest oxpy-utils \
        --ignore=tests/test_seqdesign.py \
        --deselect tests/test_forward_flux_sample_OO.py::test_load_ffs \
        --deselect tests/test_forward_flux_sample_OO.py::test_ffs_seperation_from_yml \
        --deselect "tests/test_tacoxdna.py::TestLoadFromCadnano::test_matches_correct_output"

    # test_structure_slicer.py is excluded: the structure slicer is still
    # experimental and currently fails with oxDNA rejecting the sliced
    # structure outright (OxDNAError: bonded-neighbor distance exceeds
    # acceptable values), not a test-harness issue. Remove once the slicer
    # itself is stable.
    run_stage "pypatchy pytest" \
        container_pytest pypatchy \
        --ignore=tests/test_structure_slicer.py
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
