set +e

for EXE_RUN in 1 2 3 4 5 6 7 8 9 10
do
	echo ""
 	echo ""
	echo "---> EXECUTION $EXE_RUN <---"
	python3 main.py
done

set -e
