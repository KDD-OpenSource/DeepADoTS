#!/bin/bash

set +e

for EXE_RUN in {1..5}
do
	echo ""
 	echo ""
	echo "---> EXECUTION $EXE_RUN <---"
	python main.py
done

set -e
