#!/bin/bash
# Run7 on CREMA-D. Extra args (e.g. "NUM_EPOCHS=2") are forwarded.
DATASET=cremad SCRIPT=Run7.py exec bash "$(dirname "$0")/run_server.sh" cremad Run7.py "$@"
