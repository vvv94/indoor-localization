#!/bin/bash

# Make sure that Previous Session is ended
for pid in $(ps -fu vtsouval | grep python3 | awk "{print \$2}{print \$3}"); do
  kill -9 $pid 2> /dev/null;
done;

python3 -m regression