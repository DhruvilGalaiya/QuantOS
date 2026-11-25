#!/usr/bin/env bash
# The newest data/ lives in one file on the repo's "data-latest" release, replaced on every refresh, so daily
# data never piles up in git history (about 17 MB a day if committed).
#
#   scripts/data_bundle.sh pull   # download and unpack over ./data (CI, or locally to catch up with the cloud)
#   scripts/data_bundle.sh push   # pack ./data and replace the release file (CI only: needs gh + GH_TOKEN)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${GITHUB_REPOSITORY:-DhruvilGalaiya/QuantOS}"
TAG=data-latest
FILE=quantos-data.tar.gz
TMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"

case "${1:-}" in
  pull)
    if curl -fsL -o "$TMP/$FILE" "https://github.com/$REPO/releases/download/$TAG/$FILE"; then
      tar -xzf "$TMP/$FILE" -C .
      echo "restored data/ from the $TAG release ($(du -h "$TMP/$FILE" | cut -f1))"
    else
      echo "no $TAG release yet: using the data committed to the repo"
    fi
    ;;
  push)
    # backups and the expiry-week chains are local-only; logs, locks and caches are per machine
    tar -czf "$TMP/$FILE" \
      --exclude=data/backups --exclude=data/derivatives/expiry_week \
      --exclude=data/system/logs --exclude=data/system/locks --exclude=data/system/cache \
      --exclude='*.lock' --exclude='.refresh.lock' data
    gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 || \
      gh release create "$TAG" --repo "$REPO" --prerelease --title "Latest data (automated)" \
        --notes "Newest data/ written by the daily refresh workflow. Replaced on every run; not a software release."
    gh release upload "$TAG" "$TMP/$FILE" --repo "$REPO" --clobber
    echo "uploaded $(du -h "$TMP/$FILE" | cut -f1) to the $TAG release"
    ;;
  *) echo "usage: $0 pull|push" >&2; exit 2 ;;
esac
