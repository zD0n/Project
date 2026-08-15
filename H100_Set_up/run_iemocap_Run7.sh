#!/bin/bash
# Run7 on IEMOCAP. Extra args (e.g. "NUM_EPOCHS=2") are forwarded.
DATASET=iemocap SCRIPT=Run7.py exec bash "$(dirname "$0")/run_server.sh" iemocap Run7.py "$@"
