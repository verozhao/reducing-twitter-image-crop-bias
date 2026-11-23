#!/usr/bin/env bash
# Download Twitter's released saliency program and model (Apache-2.0) into third_party/twitter/.
# Source: https://github.com/twitter-research/image-crop-analysis, pinned to one commit.
set -euo pipefail
SHA=d66ad3bb3881fdbd12a27e04c3e57cc8fd772dd1
BASE="https://raw.githubusercontent.com/twitter-research/image-crop-analysis/${SHA}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/third_party/twitter"
case "$(uname -s)" in
  Darwin) PLATFORM=mac ;;
  Linux) PLATFORM=linux ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
mkdir -p "${OUT}"
curl -fsSL -o "${OUT}/candidate_crops" "${BASE}/bin/${PLATFORM}/candidate_crops"
curl -fsSL -o "${OUT}/fastgaze.vxm" "${BASE}/bin/fastgaze.vxm"
curl -fsSL -o "${OUT}/LICENSE" "${BASE}/LICENSE"
curl -fsSL -o "${OUT}/results.txt" "${BASE}/data/results.txt"
chmod +x "${OUT}/candidate_crops"
echo "Saved to ${OUT} (${PLATFORM} build, commit ${SHA:0:7})"
