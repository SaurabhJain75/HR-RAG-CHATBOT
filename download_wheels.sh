#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# download_wheels.sh
# Downloads all packages into ./wheelhouse for offline Docker builds
#
# Run this ONCE on a machine with good internet:
#   chmod +x download_wheels.sh
#   ./download_wheels.sh
#
# Then build Docker image offline:
#   docker-compose up --build
# ══════════════════════════════════════════════════════════════════════════════

set -e  # exit on any error

echo "══════════════════════════════════════"
echo "  HR RAG Chatbot — Wheel Downloader"
echo "══════════════════════════════════════"

# ── Check pip is available ────────────────────────────────────────────────────
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Install Python first."
    exit 1
fi

# ── Check requirements.txt exists ─────────────────────────────────────────────
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found."
    echo "   Run this script from your project root folder."
    exit 1
fi

# ── Clean and create wheelhouse ───────────────────────────────────────────────
echo ""
echo "→ Cleaning old wheelhouse..."
rm -rf wheelhouse
mkdir -p wheelhouse

# ── Upgrade pip first ─────────────────────────────────────────────────────────
echo "→ Upgrading pip..."
pip install --upgrade pip -q

# ── Download all packages ─────────────────────────────────────────────────────
echo ""
echo "→ Downloading all packages into ./wheelhouse..."
echo "  This may take a few minutes depending on your connection."
echo ""

pip download \
  -r requirements.txt \
  -d ./wheelhouse \
  --prefer-binary

# ── Also download pip itself (needed inside Docker) ───────────────────────────
pip download pip -d ./wheelhouse --prefer-binary -q

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo "  Download Complete!"
echo "══════════════════════════════════════"
echo ""
echo "  Files downloaded : $(ls wheelhouse/ | wc -l)"
echo "  Total size       : $(du -sh wheelhouse/ | cut -f1)"
echo ""
echo "  To install offline:"
echo "  pip install --no-index --find-links ./wheelhouse -r requirements.txt"
echo ""
echo "  To build Docker offline:"
echo "  docker-compose up --build"
echo "══════════════════════════════════════"
