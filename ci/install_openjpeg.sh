#!/bin/sh

# Bail out if any errors arise.
set -e

# Compile openjpeg from source, install into home directory.
wget https://github.com/uclouvain/openjpeg/archive/v2.1.1.tar.gz
tar xvfz v2.1.1.tar.gz
mkdir -p openjpeg-2.1.1/build
cd openjpeg-2.1.1/build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/openjpeg && make && make install

# Setup the configuration file. 
mkdir -p $HOME/.config/glymur
cat << EOF > $HOME/.config/glymur/glymurrc
[library]
openjp2:  $HOME/openjpeg/lib/libopenjp2.so
EOF
