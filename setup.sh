#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Python
if command_exists python3; then
    echo "Python already installed: $(python3 --version)"
elif command_exists python; then
    echo "Python already installed: $(python --version)"
else
    echo "Python 3 not found; install it via your package manager (e.g. apt install python3)."
    exit 1
fi

# C compiler
if command_exists cc || command_exists clang || command_exists gcc; then
    echo "C compiler already installed."
else
    echo "C compiler not found; install clang or gcc via your package manager."
    exit 1
fi

# ext/ dependencies
if [ ! -f "ext/minifb/include/MiniFB.h" ]; then
    echo "ext/minifb missing; copying from apps/.lib99/dskbuf..."
    LIB99="$(dirname "$SCRIPT_DIR")/.lib99/dskbuf"
    if [ -d "$LIB99" ]; then
        mkdir -p "ext/minifb"
        cp -r "$LIB99/." "ext/minifb/"
    else
        echo "WARNING: apps/.lib99/dskbuf not found; ext/minifb must be provided manually."
    fi
fi

# Runtime-id output dir (build/<rid>), matching make.py.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) rid_os="win" ;;
    Darwin*)              rid_os="osx" ;;
    *)                    rid_os="linux" ;;
esac
case "$(uname -m)" in
    x86_64|amd64) rid_arch="x64" ;;
    aarch64|arm64) rid_arch="arm64" ;;
    *)             rid_arch="x86" ;;
esac
mkdir -p "build/$rid_os-$rid_arch/shots" docs/shots
echo "rpg99 build tools are ready."