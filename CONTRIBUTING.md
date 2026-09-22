# Contributing to PyPatchy

## Branching and pull requests

- `main` is the trunk. `doc` is a separate branch for documentation work (see its own scope below).
- Both branches are protected: nobody pushes directly to `main` or `doc`, including maintainers. All changes land via a pull request.
- Open a feature/fix branch off `main` (e.g. `feature/short-description` or `fix/short-description`), keep it short-lived, and delete it once the PR merges.
- PRs into `main` or `doc` need at least **1 approving review** before merging.
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

This repo does not include the SAT-based crystal design subsystem (`pypatchy/design/`) or any dependency on the private `polycubes-clone` (`libtlm`) library — those aren't part of the public codebase. If your contribution needs either, it's likely out of scope here.
