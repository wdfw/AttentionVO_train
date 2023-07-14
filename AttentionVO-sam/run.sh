#!/bin/bash

cd /app
source "/opt/ros/$ROS_DISTRO/setup.bash"

# MODEL=/drone/AttentionVO/models/tartanvo_1914.pkl
# MODEL=/media/sam/本機磁碟/tartanvo/model/only_postnet_5000_without_shuffle_Adam/tartanvo_20230603_1733_4925.pkl
MODEL=/media/sam/本機磁碟/tartanvo/model/postnet_1000_wo-scale_Adam/tartanvo_20230606_0248_993.pkl
RVIZ_CONFIG=./config.rviz


# Enable tracing
set -x

GT_DIR=""
while getopts "ked:g:rco" opt
do
  case $opt in
    # VAR=$OPTARG
    k)
        DATASET_FORMAT=--kitti
        ;;
    e)
        DATASET_FORMAT=--euroc
        ;;
    d)
        DATASET_DIR=$OPTARG
        ;;
    g)  
        GT_DIR="--pose-file ${OPTARG}"
        ;;
    r)
        python3 /drone/AttentionVO/vo_trajectory_from_folder.py \
            --model-name ${MODEL} \
            ${DATASET_FORMAT} \
            --batch-size 1 --worker-num 1 \
            --test-dir ${DATASET_DIR} ${GT_DIR}  \
            # --save-flow
        ;;
    c)
        python3 /drone/AttentionVO/vo_trajectory_from_folder_onlypose.py \
            --model-name ${MODEL} \
            ${DATASET_FORMAT} \
            --batch-size 1 --worker-num 1 \
            --test-dir ${DATASET_DIR} ${GT_DIR} 
            
        ;;
    o)
        rosparam set /img_dir ${DATASET_DIR}
        python3 /drone/AttentionVO/tartanvo_node.py \
        & rosrun rviz rviz -d /drone/AttentionVO/config.rviz
        # python3 tartanvo_node.py \
        
        ;;
    ?) 
        echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    *)
        echo "*"
        ;;
  esac
done

# Disable tracing
set +x
