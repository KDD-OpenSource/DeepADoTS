#!/bin/bash
source ./venv/bin/activate

set +e

for EXE_RUN in {1..5}
do
	echo ""
 	echo ""
	echo "---> EXECUTION $EXE_RUN <---"
	python3 ./main.py
done

set -e
