# This script is for downloading the datasets given in Part 0: Datasets & Setup
# It is also for setting up the entire file structure as well
#!/bin/bash

mkdir -p zip 
cd zip 

echo "Downloading EB-NeRD datasets"
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip

echo "Downloading MIND datasets"
#Would do this but getting some 401 error so had to modify it to use HF Token instead.
#I set it as an environment variable because safety
# wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip
# wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip

if [ -z "$HF_TOKEN" ]; then
    echo "Need to export HF_TOKEN="HF_..." first"
    exit 1
fi

curl -L -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip" -o MINDsmall_train.zip
curl -L -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip" -o MINDsmall_dev.zip

cd ..
mkdir -p data

echo "Extracting zip files"
unzip ./zip/ebnerd_demo.zip -d ./data/ebnerd_demo
unzip ./zip/ebnerd_small.zip -d ./data/ebnerd_small
unzip ./zip/Ekstra_Bladet_word2vec.zip -d ./data/Ekstra_Bladet_word2vec
unzip ./zip/google_bert_base_multilingual_cased.zip -d ./data/google_bert_base_multilingual_cased
unzip ./zip/MINDsmall_train.zip -d ./data
unzip ./zip/MINDsmall_dev.zip -d ./data

#Datafiles once schema is applied
mkdir -p processed_data

#Datafiles once split into train/val/test
mkdir -p split_data
