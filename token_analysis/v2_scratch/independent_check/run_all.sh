#!/bin/bash
P=../../../.venv/bin/python
for m in g12 g27 l70 q7; do for k in opi opistart opimiddle opihistory; do $P run_main.py $k $m orig,strict; done; done
for m in g27 l70; do $P run_main.py liars $m orig,block; done
for m in g12 g27 l70 q7; do $P run_main.py tt $m orig,block; $P run_main.py taboo20 $m orig; $P run_main.py taboo $m orig; done
