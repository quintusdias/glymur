#!/bin/sh
# Install openjpeg library version 1.5

# Bail out if any errors arise.
set -e

# Compile openjpeg from source, install into home directory.
wget https://github.com/uclouvain/openjpeg/archive/version.1.5.2.tar.gz
tar xvfz version.1.5.2.tar.gz
mkdir -p openjpeg-version.1.5.2/build
cd openjpeg-version.1.5.2/build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/openjpeg && make && make install

# Setup the configuration file. 
mkdir -p $HOME/.config/glymur
cat << EOF > $HOME/.config/glymur/glymurrc
[library]
openjpeg:  $HOME/openjpeg/lib/libopenjpeg.so
EOF
