#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SC=/tmp/claude-10003/-home-sam-benny-external-anubatechnologies-com/6de6c3a2-577b-4636-8e47-34c6a1dd5048/scratchpad
# wait for the builder python to exit
while ps -eo args | grep -q "[.]venv/bin/python -$"; do sleep 30; done
sleep 5
echo "=== build tail ==="; tail -4 "$SC/engine_build.log"
echo; echo "=== artifacts ==="; ls -lh rf_trained/ | awk 'NR>1{print "  ",$5,$9,$10}'
