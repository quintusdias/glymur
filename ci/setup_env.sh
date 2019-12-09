#!/bin/bash -e

MINICONDA_DIR="$HOME/miniconda3"
if [ -d "$MINICONDA_DIR" ]; then
    echo
    echo "rm -rf "$MINICONDA_DIR""
    rm -rf "$MINICONDA_DIR"
fi
echo "set MINICONDA_DIR to ""$MINICONDA_DIR"

UNAME_OS=$(uname)
if [[ "$UNAME_OS" == 'Linux' ]]; then
    CONDA_OS="Linux-x86_64"
elif [[ "$UNAME_OS" == 'Darwin' ]]; then
    CONDA_OS="MacOSX-x86_64"
else
  echo "OS $UNAME_OS not supported"
  exit 1
fi
echo "set CONDA_OS to ""$CONDA_OS"

echo "Install Miniconda"
wget -q "https://repo.continuum.io/miniconda/Miniconda3-latest-$CONDA_OS.sh" -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b

export PATH=$MINICONDA_DIR/bin:$PATH

echo
echo "which conda"
which conda

echo
echo "update conda"
conda config --set ssl_verify false
conda config --set quiet true --set always_yes true --set changeps1 false
conda update -n base conda

echo "conda info -a"
conda info -a

echo "source deactivate"
source deactivate

echo "conda list (root environment)"
conda list

# Clean up any left-over from a previous build
# (note workaround for https://github.com/conda/conda/issues/2679:
#  `conda env remove` issue)
conda remove --all -q -y -n glymur

echo
echo "conda env create -q --file=${ENV_FILE}"
time conda env create -q --file="${ENV_FILE}"


echo "conda activate glymur"
conda activate glymur

# Make sure any error below is reported as such

echo "[Build me]"
python setup.py install

# XXX: Some of our environments end up with old verisons of pip (10.x)
# Adding a new enough verison of pip to the requirements explodes the
# solve time. Just using pip to update itself.
echo "[Updating pip]"
python -m pip install --no-deps -U pip wheel setuptools

echo "[Install glymur]"
python -m pip install --no-build-isolation -e .

echo
echo "conda list"
conda list

echo "done"

