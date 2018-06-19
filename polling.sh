# Poll the current branch every hour, execute the complete pipeline and send the results to Slack
# Needs the environment variable TOKEN to be set

# Name of the virtual environment
VENV_DIR="venv_mp"

while true
do
  date +%d.%m.%Y/%H-%M-%S
  COMMIT="$(git log -1 | tr '\n' ',' | awk '{split($0,a,",,"); print a[2]}' | awk '{split($0,a,"    ,    "); print a[1]}')"
  echo "Current commit: $COMMIT"

  GIT_PULL_OUTCOME="$(git pull)"
  echo "$GIT_PULL_OUTCOME"
  UPTODATE='Already up to date.'
  IS_UPTODATE=$(expr "${GIT_PULL_OUTCOME}" == "${UPTODATE}")

  if [ "$IS_UPTODATE" -eq 1 ]
  then
    echo "No changes detected"
  else
    echo "Changes detected"
    source $VENV_DIR/bin/activate
    # Forward stderr and stdout
    python main.py > program_output 2>&1
    EXIT="$?"
    cat program_output | grep "Traceback" -A 20 > exceptions
    N_EXCEPTIONS=$(cat exceptions | grep "Traceback" | wc -l)
    AUTHOR=$(git log -1 | tr '\n' ',' | awk '{split($0,a,"Author: "); print a[2]}' | awk '{split($0,a," <"); print a[1]}')
    if [ "$N_EXCEPTIONS" -eq 0 ] && [ "$EXIT" -eq 0 ]
    then
      MSG="$COMMIT ($AUTHOR): All good!"
    else
      MSG="$COMMIT ($AUTHOR): Exit code $EXIT, $N_EXCEPTIONS exceptions: $(cat exceptions)"
    fi
    echo "$MSG"
    curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$MSG\"}" "https://hooks.slack.com/services/TA9LKQT8D/BBAD4G01L/$TOKEN"
  fi
  sleep 60m
done
