# Recommended: Create a notebook config for jupyter

printf "\n\nStart Notebook\n" >> jupyter.logs;

source activate johenv;

jupyter notebook &>> jupyter.logs &

echo Jupyter process id: $! >> jupyter.logs;

tail -n 3 -f jupyter.logs;

# source activate johenv && jupyter notebook --NotebookApp.token='' --port 9999
