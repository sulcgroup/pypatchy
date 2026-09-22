# Contributing to PyPatchy

## Access and branching

This repo is private. Access is granted per person, not by branch protection rules:

- **Write access** (a small group of trusted maintainers) can push branches and merge directly into `main`/`doc`.
- **Read access** (everyone else with access to the repo) can view and clone it, but can't push to it directly — contribute by **forking** the repo, branching and committing on your fork, and opening a PR from your fork back into `sulcgroup/pypatchy`. A maintainer reviews and merges it.

`main` is the trunk. `doc` is a separate branch for documentation work (see its own scope below).

- Open a feature/fix branch off `main` (e.g. `feature/short-description` or `fix/short-description`), keep it short-lived, and delete it once the PR merges.
- Prefer several small, focused PRs over one large one — easier to review, easier to revert if something's wrong.

## Before opening a PR

- Run the relevant test suite locally:
  ```bash
  pytest tests/
  ```
  See this repo's `CLAUDE.md` for which test files are considered stable vs. still WIP.
- If your change touches the Docker image (`Dockerfile`, `dock.sh`), a full local build via `./dock.sh` is expected to pass before requesting review — it's slow, but it's the closest thing this repo has to CI right now.
- Keep the PR description focused on *why*, not just *what* — the diff already shows what changed.

## Reporting issues

- Use GitHub Issues. Include: what you expected, what happened instead, and enough to reproduce (rule string, ensemble config, or a minimal script).

## Scope note

This repo does not include the SAT-based crystal design subsystem (`pypatchy/design/`) or any dependency on the private `polycubes-clone` (`libtlm`) library — those aren't part of this shared codebase. If your contribution needs either, it's likely out of scope here.
