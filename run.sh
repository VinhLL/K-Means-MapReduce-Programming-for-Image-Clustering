#!/usr/bin/env bash

result_folder="./dataset/result"  
output_folder="./dataset/result" 
mask_folder="./dataset/mask"  

if [ -d "$result_folder" ]; then
    echo "Đang xóa thư mục kết quả hiện tại: $result_folder"
    rm -r -f "$result_folder"
fi

if [ -d "$output_folder" ]; then
    echo "Đang xóa thư mục đầu ra hiện tại: $output_folder"
    rm -r -f "$output_folder" 
fi

mkdir -p $output_folder $result_folder

for file in ./dataset/input/*; do
    echo "Đang xử lý file: $(basename "$file")"
    rm -r -f ./tmp/* 

    mkdir ./tmp  
    python3 data_prep.py --src_img "$file" --dst_folder ./tmp --k_init_centroids 3 

    hdfs dfs -rm -r -f /KMeans/Input
    hdfs dfs -rm -r -f /KMeans/Output

    hdfs dfs -mkdir -p /KMeans/Input
    hdfs dfs -mkdir -p /KMeans/Output

    hdfs dfs -put ./tmp/points.txt ./tmp/clusters.txt /KMeans/Input/

    hdfs dfs -rm -r -f /KMeans/Output/*

    # Đặt các tham số cho chương trình MapReduce KMeans
    JAR_PATH=./kmeans_mapreduce.jar  
    MAIN_CLASS=Main  
    INPUT_FILE_PATH=/KMeans/Input/points.txt  
    STATE_PATH=/KMeans/Input/clusters.txt  
    NUMBER_OF_REDUCERS=1  
    OUTPUT_DIR=/KMeans/Output
    DELTA=1000000000.0  
    MAX_ITERATIONS=30 
    DISTANCE=eucl  

    hadoop jar ${JAR_PATH} ${MAIN_CLASS} --input ${INPUT_FILE_PATH} \
    --state ${STATE_PATH} \
    --number ${NUMBER_OF_REDUCERS} \
    --output ${OUTPUT_DIR} \
    --delta ${DELTA} \
    --max ${MAX_ITERATIONS} \
    --distance ${DISTANCE}

    LAST_DIR="$(hdfs dfs -ls -t -C /KMeans/Output | head -1)" 
    hdfs dfs -get "$LAST_DIR/part-r-00000" ./tmp 
    mv ./tmp/part-r-00000 ./tmp/kmeans-output.txt 

    python3 visualize_results.py --clusters_path ./tmp/kmeans-output.txt \
                                  --src_img ./tmp/segmented_image.png \
                                  --dst_img "$result_folder/$(basename "$file")"

    rm -r -f ./tmp  

done

# So sánh kết quả phân nhóm với các ảnh mask
python3 compare_image.py --result_folder "$result_folder" --mask_folder "$mask_folder"
