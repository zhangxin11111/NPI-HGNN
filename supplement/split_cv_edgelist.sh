#!/bin/bash
echo $(date +%F%n%T)
python generate_edgelist.py --dataset $1 --samplingType $2 & pid=( $! )
echo python generate_edgelist.py --dataset $1 --samplingType $2
echo pid: $pid
PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
while [ $PID_EXIST ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
done
echo pid"("$pid")" is finished

