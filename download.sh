#!/bin/bash

base_dir=$(cd "$(dirname "$0")" && pwd)

# wikipedia contents
data_type=passages-c400-jawiki-20230403
model_type=multilingual-e5-base-passage
data_dir="${base_dir}/dataset/${data_type}"
num_of_docs=5555583
data_files="
0000.parquet
0001.parquet
0002.parquet
0003.parquet
0004.parquet
0005.parquet
0006.parquet
0007.parquet
"

mkdir -p "${data_dir}"

for data_file in ${data_files} ; do
  if [[ ! -f "${data_dir}/${data_file}" ]] ; then
    echo -n "Downloading ${data_file}... "
    curl -sL -o "${data_dir}/${data_file}" \
      "https://huggingface.co/datasets/singletongue/wikipedia-utils/resolve/refs%2Fconvert%2Fparquet/${data_type}/train/${data_file}?download=true" || exit 1
    echo "[OK]"
  fi
done

mkdir -p "${data_dir}/${model_type}"

count=0
while [[ ${count} -lt ${num_of_docs} ]] ; do
  data_file="${data_dir}/${model_type}/${count}.npz"
  if [[ ! -f "${data_file}" ]] ; then
    echo -n "Downloading ${count}.npz... "
    curl -sL -o "${data_file}" \
      "https://huggingface.co/datasets/hotchpotch/wikipedia-passages-jawiki-embeddings/resolve/main/embs/${data_type}/${model_type}/${count}.npz" || exit 1
    echo "[OK]"
  fi
  count=$((count + 100000))
done

