#!/bin/bash
# 2-epoch smoke test on IEMOCAP.
exec bash "$(dirname "$0")/run_server.sh" iemocap Run7.py NUM_EPOCHS=2 "$@"
