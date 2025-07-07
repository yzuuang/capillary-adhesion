directory=venv
if [ -d "${directory}" ]; then
    rm -rf "${directory}"
fi
python3.12 -m venv "${directory}"
source "${directory}"/bin/activate
pip install --upgrade pip
pip install 'flit_core >=3.8.0,<4' meson-python
pip install -e .[test]
pip install SurfaceTopography
pip install --force-reinstall --no-build-isolation -e ${HOME}/muGrid
