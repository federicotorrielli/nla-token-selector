#!/bin/bash
# Thin wrapper: launch an SGLang server in the active Python env, exporting the
# CUDA paths SGLang's kernels need. Set PENV to a conda env prefix, or rely on an
# already-activated environment ($CONDA_PREFIX). All args pass through to
# `python -m sglang.launch_server`.
PENV=${PENV:-${CONDA_PREFIX:?activate your env or set PENV to its prefix}}
export CUDA_HOME=$PENV CUDA_PATH=$PENV
export PATH=$PENV/bin:$PATH
export LIBRARY_PATH=$PENV/lib:$PENV/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PENV/lib:$PENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CPATH=$PENV/include:$PENV/targets/x86_64-linux/include:$CPATH
[ -e "$PENV/lib64" ] || ln -s "$PENV/lib" "$PENV/lib64"
exec "$PENV/bin/python" -m sglang.launch_server "$@"
