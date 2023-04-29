#!/bin/bash
echo $(date +%F%n%T)
python generate_kmer.py --dataset $1 & pid=( $! )
echo python generate_kmer.py --dataset $1
echo pid: $pid
PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
while [ $PID_EXIST ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
done
echo pid"("$pid")" is finished

python generate_frequency.py --dataset $1 & pid=( $! )
echo python generate_frequency.py --dataset $1
echo pid: $pid
PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
while [ $PID_EXIST ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
done
echo pid"("$pid")" is finished

python generate_pyfeat.py --dataset $1 & pid=( $! )
echo python generate_pyfeat.py --dataset $1
echo pid: $pid
PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
while [ $PID_EXIST ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
done
echo pid"("$pid")" is finished

