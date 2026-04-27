#!/bin/bash
# Run from project root.
set -e

INPUT="${1:-hdfs:///dic_shared/amazon-reviews/full/reviews_devset.json}"
OUTPUT_DIR="${2:-hdfs:///user/$(whoami)/a1_out}"

HADOOP_STREAMING_JAR="/usr/lib/hadoop/tools/lib/hadoop-streaming.jar"


python src/MRjobs.py -r hadoop \
    --hadoop-streaming-jar "$HADOOP_STREAMING_JAR" \
    "$INPUT" \
    --output-dir "$OUTPUT_DIR"

# Merge HDFS part-files into a single output.txt
hadoop fs -getmerge "$OUTPUT_DIR" output.txt
echo "Wrote output.txt"