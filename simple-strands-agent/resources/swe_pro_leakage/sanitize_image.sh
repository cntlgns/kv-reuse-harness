#!/bin/bash
# Runs INSIDE the container. Strips every git ref except HEAD's own history
set -euo pipefail

GIT_ROOT="/app"
if [ ! -d "$GIT_ROOT/.git" ]; then
  echo "NO_GIT_FOUND" >&2
  exit 0
fi

cd "$GIT_ROOT"
HEAD_SHA=$(git rev-parse HEAD)
echo "git root: $GIT_ROOT, HEAD: $HEAD_SHA"

# Drop remotes FIRST
for r in $(git remote); do git remote remove "$r"; done

# Delete every remaining ref (local branches, tags, stash, notes, replace).
git for-each-ref --format='%(refname) %(symref)' \
  | awk '$2=="" {print "delete " $1}' \
  | git update-ref --stdin

# Re-create a single local branch at HEAD and point HEAD at it.
git checkout -B main "$HEAD_SHA" >/dev/null 2>&1

# Wipe reflog & GC unreachable objects. Need --aggressive for full repack.
git reflog expire --expire=now --all
git gc --prune=now --aggressive >/dev/null 2>&1

AFTER_REFS=$(git for-each-ref | wc -l)
AFTER_SIZE=$(du -sb .git | awk '{print $1}')

# Sanity: HEAD must still resolve to the same commit.
NEW_HEAD=$(git rev-parse HEAD)
if [ "$NEW_HEAD" != "$HEAD_SHA" ]; then
  echo "ERROR: HEAD changed during cleanup ($HEAD_SHA -> $NEW_HEAD)" >&2
  exit 1
fi
