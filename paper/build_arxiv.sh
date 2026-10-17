#!/usr/bin/env bash
# Build the arXiv upload: LaTeX source plus the figures it includes.
#
#   ./build_arxiv.sh            -> arxiv/ (tree) and arxiv.tar.gz (upload this)
#
# arXiv does not accept Markdown as source. This dumps the pandoc-generated
# LaTeX with the same layout options as the PDF build, rewrites the figure
# paths to a flat figures/ directory, and tars the result.
set -euo pipefail
cd "$(dirname "$0")"

out=arxiv
rm -rf "$out" arxiv.tar.gz
mkdir -p "$out/figures"

pandoc main.md -s -o "$out/main.tex" -V geometry:margin=1in -V fontsize=11pt
sed -i 's#\.\./figures/#figures/#g' "$out/main.tex"
cp ../figures/*.pdf "$out/figures/"

tar -czf arxiv.tar.gz -C "$out" main.tex figures
echo "wrote $(pwd)/arxiv.tar.gz"
