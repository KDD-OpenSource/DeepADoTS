#!/bin/bash

set +e

for EXE_RUN in {1..10}
do
	echo ""
 	echo ""
	echo "---> EXECUTION $EXE_RUN <---"
	python3 main.py
done

set -e
