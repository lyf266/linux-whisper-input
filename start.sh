#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Activate Python virtual environment
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi

# Dynamically construct LD_LIBRARY_PATH for pip-installed nvidia dependencies
NVIDIA_DIRS=$(python -c '
import os, sys, glob
venv_path = sys.prefix
paths = glob.glob(os.path.join(venv_path, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
print(":".join(paths))
' 2>/dev/null || echo "")

if [ -n "$NVIDIA_DIRS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_DIRS:${LD_LIBRARY_PATH:-}"
fi

# Execute daemon (using exec to let systemd manage the python process directly)
exec python -u "$DIR/daemon.py"
