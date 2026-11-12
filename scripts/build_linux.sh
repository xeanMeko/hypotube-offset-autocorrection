#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Build the Linux bundle on Linux." >&2
    exit 1
fi

build_python="${MEASURE_SHIFT_BUILD_PYTHON:-/usr/bin/python3}"
venv_dir="$project_root/build/linux-venv"
if [[ "${1:-}" != "--skip-install" ]]; then
    "$build_python" -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install --cache-dir "$project_root/build/pip-cache" \
        '.[cameras,motion]' 'pyinstaller>=6,<7'
fi

mkdir -p build/linux-spec
XDG_CACHE_HOME="$project_root/build/cache" "$venv_dir/bin/python" -m PyInstaller \
    --noconfirm \
    --clean \
    --onedir \
    --name measure-shift \
    --distpath dist \
    --workpath build/pyinstaller-linux \
    --specpath build/linux-spec \
    --paths src \
    --collect-all pypylon \
    --collect-all pyueye \
    --collect-submodules tinkerforge \
    "$venv_dir/bin/measure-shift"

"$venv_dir/bin/python" -m pip freeze > dist/measure-shift/build-requirements.txt
tar -czf "dist/measure-shift-linux-$(uname -m).tar.gz" -C dist measure-shift

echo "Run: $project_root/dist/measure-shift/measure-shift --demo"
echo "Archive: $project_root/dist/measure-shift-linux-$(uname -m).tar.gz"
