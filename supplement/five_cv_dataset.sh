#!/bin/bash
echo $(date +%F%n%T)
i=0
python generate_node_vec.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i & pid=( $! )
echo python generate_node_vec.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i
echo pid: $pid
let "i = $i + 1"
while [ "$i" -lt 5 ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
    if [ ! $PID_EXIST ]; then
        echo pid"("$pid")" is finished
		python generate_node_vec.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i & pid=( $! )
		echo python generate_node_vec.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i
		echo pid: $pid
        let "i = $i + 1"
    fi
done

i=0
while [ "$i" -lt 5 ]
do
    sleep 1s
    PID_EXIST=$(ps -ef | grep NPHN | awk '{print $2}'| grep -w $pid)
    if [ ! $PID_EXIST ]; then
        echo pid"("$pid")" is finished
        python generate_dataset.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i & pid=( $! )
        echo generate_dataset.py --dataset $1 --nodeVecType $2 --subgraph_type $3 --samplingType $4 --fold $i "&"
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

