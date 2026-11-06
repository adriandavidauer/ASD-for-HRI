#!/usr/bin/env bash
# Commit the README badges written by update_badges.py and push them to the checked-out branch.
#
# This cannot loop: pushes made with GITHUB_TOKEN never trigger workflow runs, and the
# [skip ci] in the message is a second guard in case the token is ever swapped for a PAT.
# If the branch moved during the run (e.g. the other badge workflow pushed first), the badges
# are rebuilt on top of the new head and pushed once more. Rebuilding instead of rebasing avoids
# conflicts between the adjacent badge lines. Trade-off: the rebuilt badges carry this run's
# numbers on top of a newer commit; that commit's own run corrects them shortly after.
# Expects the env vars of update_badges.py.
set -euo pipefail

BRANCH=${GITHUB_REF#refs/heads/}
git config user.name "github-actions[bot]"
git config user.email "41898333+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2; do
  if git diff --quiet README.md; then
    echo "Badges unchanged"; exit 0
  fi
  git commit README.md -m "Update README badges [skip ci]"
  if git push origin "HEAD:$BRANCH"; then
    exit 0
  fi
  echo "Push rejected (attempt $attempt); rebuilding the badges on the new head of $BRANCH"
  git fetch origin "$BRANCH"
  git reset --hard "origin/$BRANCH"
  python .github/scripts/update_badges.py
done
echo "::warning::$BRANCH kept moving during the run; the next run will refresh the badges."
