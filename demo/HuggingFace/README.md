# TRT Compilation of HF GPT-XL & Larger Models with 1024 Seq Length

### Instance  / Repo

P4 Instance / A100-SXM4-40GB
```
mkdir workspace && cd workspace
git clone https://github.com/dhanainme/TensorRT/
```


### Docker Env

Build DockerFile from TensorRT Repo - Branch release/8.2 - Ubuntu 20.4 based TRT Container as `tensorrt-ubuntu20.04-cuda11.4:latest`
The build the Dockerfile from `TensorRT/demo/HuggingFace/Dockerfile` 


### Running compilation
```
docker run -it --gpus "device=1" -v /home/ubuntu/workspace:/workspace/ 73e2cb1e5b6b /bin/bash

# Inside the docker container
cd /workspace/TensorRT/demo/HuggingFace
mkdir logs

export EXPERIMENT_NAME=gpt_xl_1024seq_5_batch_size
python3 -u compile.py 2>&1 | tee logs/$EXPERIMENT_NAME.logs
```

# Experiments

|Run1 | Default Configuration. Batch 5 / 1024 / GPT-XL.| We expect this to compile. But have issues with inference. Garbage. If this works this is a more representative 1 for running benchmark tests.|
|Run2 | BATCH_SIZE=1. Only changes the BATCH_SIZE from Run1 | Compare this with Run1 for GPU Usage |
|Run3 | INIT_MODEL_FROM_CONFIG=1, BATCH_SIZE=1 |10GB Model. We expect compilation to fail|
|Run4 | BATCH_SIZE=1, MAX_SEQ_LENGTH=64 |  Batch 5 / 64 / GPT-XL| Has worked before. The claimed 6x speedup is from this config.|


