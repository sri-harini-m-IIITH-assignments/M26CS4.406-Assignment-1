# This script is for downloading the datasets given in Part 0: Datasets & Setup
# It is also for setting up the entire file structure as well
#!/bin/bash

set -e
mkdir -p zip data processed_data split_data

download () { 
    local url=$1
    local output=$2
    local auto_extract=$3

    if [ -f "$output" ]; then
        echo "$output already exists, skipping download."
    else
        echo "Downloading $url to $output"
        if [ -n "$auth_header" ]; then
            curl -L -H "$auth_header" "$url" -o "$output"
        else
            wget -O "$output" "$url"
        fi
    fi
}

extract () {
    local file=$1
    local dest_dir=$2

    if [ -d "$dest_dir" ]; then
        echo "$dest_dir already exists, skipping extraction."
    else
        echo "Extracting $file to $dest_dir"
        unzip "$file" -d "$dest_dir"
    fi
}

echo "Downloading EB-NeRD datasets"
download "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip" "./zip/ebnerd_demo.zip"
download "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip" "./zip/ebnerd_small.zip"
download "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip" "./zip/Ekstra_Bladet_word2vec.zip"
download "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip" "./zip/google_bert_base_multilingual_cased.zip"

echo "Downloading MIND datasets"
#Would do this but getting some 401 error so had to modify it to use HF Token instead.
#I set it as an environment variable because safety
# wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip
# wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip

# if [ -z "$HF_TOKEN" ]; then
#     echo "Need to export HF_TOKEN="HF_..." first"
#     exit 1
# fi

# auth_header="Authorization: Bearer $HF_TOKEN"

# download "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip" "./zip/MINDsmall_train.zip"
# download "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip" "./zip/MINDsmall_dev.zip"

download "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip" "./zip/MINDsmall_train.zip"
download "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip" "./zip/MINDsmall_dev.zip"

echo "Extracting zip files"
extract "./zip/ebnerd_demo.zip" "./data/ebnerd_demo"
extract "./zip/ebnerd_small.zip" "./data/ebnerd_small"
extract "./zip/Ekstra_Bladet_word2vec.zip" "./data/Ekstra_Bladet_word2vec"
extract "./zip/google_bert_base_multilingual_cased.zip" "./data/google_bert_base_multilingual_cased"
extract "./zip/MINDsmall_train.zip" "./data/MINDsmall_train"
extract "./zip/MINDsmall_dev.zip" "./data/MINDsmall_dev"

echo "All datasets downloaded and extracted successfully."
