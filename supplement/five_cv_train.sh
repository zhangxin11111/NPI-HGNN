#!/bin/bash
echo $(date +%F%n%T)
i=0
python train.py --dataset $1 --nodeVecType $2  --subgraph_type $3 --samplingType $4 --model_code $5 --cuda_code $6 --fold $i & pid=( $! )
echo python train.py --dataset $1 --nodeVecType $2  --subgraph_type $3 --samplingType $4 --model_code $5 --cuda_code $6 --fold $i
echo pid: $pid
let "i = $i + 1"
while [ "$i" -lt 5 ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
    if [ ! $PID_EXIST ]; then
        echo pid"("$pid")" is finished
        python train.py --dataset $1 --nodeVecType $2  --subgraph_type $3 --samplingType $4 --model_code $5 --cuda_code $6 --fold $i & pid=( $! )
	echo python train.py --dataset $1 --nodeVecType $2  --subgraph_type $3 --samplingType $4 --model_code $5 --cuda_code $6 --fold $i
        echo pid: $pid
        let "i = $i + 1"
    fi
done
while [ $PID_EXIST ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
done
echo pid"("$pid")" is finished

