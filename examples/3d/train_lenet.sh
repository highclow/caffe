#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/3Dexample/solver.prototxt $@
