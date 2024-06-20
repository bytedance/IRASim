#!/bin/bash

download_file() {
  local url=$1
  local file_name=$(basename "$url")
  
  echo "Downloading $file_name..."
  curl -O "$url"
  
  if [[ $? -ne 0 ]]; then
    echo "Failed to download $file_name"
  else
    echo "Successfully downloaded $file_name"
  fi
}

# rt-1
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/rt1_train_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/rt1_evaluation_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/rt1_checkpoints_data.tar.gz"

# bridge
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_evaluation_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_checkpoints_data.tar.gz"

# language-table
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/languagetable_train_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/languagetable_evaluation_data.tar.gz"
download_file "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/languagetable_checkpoints_data.tar.gz"
