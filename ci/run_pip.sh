#!/bin/bash

set -x

REQ="ci/requirements-${TRAVIS_PYTHON_VERSION}${JOB_TAG}.txt"

pip install --upgrade -r $REQ

