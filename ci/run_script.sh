#!/bin/bash -e


MINICONDA_DIR="$HOME/miniconda3"
echo "Set MINICONDA_DIR to ""$MINICONDA_DIR"

export PATH=$MINICONDA_DIR/bin:$PATH

source activate glymur

echo
echo "which conda"
which conda

echo "python -m unittest discover"
python -m unittest discover

python -c "import glymur; print(glymur.version.info)"
