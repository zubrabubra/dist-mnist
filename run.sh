#!/bin/bash


echo "CLUSTER_CONFIG: ${CLUSTER_CONFIG}"
echo "POD_NAME: ${POD_NAME}"

echo "-------------------------------------------"

PS_HOSTS="$(echo ${CLUSTER_CONFIG} | tr -d '\n ' | sed -e 's#,]#]#g' -e 's#,}#}#g' | jq --raw-output '.ps[]' | tr '\n' ',' | sed 's/.$//' )"
WORKER_HOSTS="$(echo ${CLUSTER_CONFIG} | tr -d '\n ' | sed -e 's#,]#]#g' -e 's#,}#}#g' | jq --raw-output '.worker[]' | tr '\n' ',' | sed 's/.$//' )"
TASK_ID=$(echo ${POD_NAME} | cut -f2 -d'-')
JOB_NAME=$(echo ${POD_NAME} | cut -f1 -d'-')

echo "CLUSTER_CONFIG: ${CLUSTER_CONFIG}"
echo "WORKER_HOSTS: ${WORKER_HOSTS}"
echo "TASK_ID: ${TASK_ID}"
echo "JOB_NAME: ${JOB_NAME}"

echo "-------------------------------------------"

if [ "$JOB_NAME" == "ps" ]; then

    echo "Starting Tensorflow parameters server"

    python /mnist.py --job_name ps --input_data_dir "${DATA_DIR}input_data" --log_dir "${DATA_DIR}logs/fully_connected_feed"

elif [ "$JOB_NAME" == "worker" ]; then

    if [ -z "$TASK_ID" ]; then
        TASK_ID=0
    fi

    echo "Starting Tensorflow worker $TASK_ID"

    python /mnist.py --job_name worker --task_index "$TASK_ID"  --input_data_dir "${DATA_DIR}input_data" --log_dir "${DATA_DIR}logs/fully_connected_feed"

fi

