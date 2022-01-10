# TRT Compilation of HF GPT-XL & Larger Models

### Instance  / Repo

P4 Instance / A100-SXM4-40GB
```
mkdir workspace && cd workspace
git clone https://github.com/dhanainme/TensorRT/
```


### Docker Env

* Build DockerFile from TensorRT Repo - Branch `release/8.2` - Ubuntu 20.4 based TRT Container as `tensorrt-ubuntu20.04-cuda11.4:latest`
* Then build the Dockerfile from `TensorRT/demo/HuggingFace/Dockerfile` 


### Running compilation
```
docker run -it --gpus "device=1" -v /home/ubuntu/workspace:/workspace/ 73e2cb1e5b6b /bin/bash

# Inside the docker container
cd /workspace/TensorRT/demo/HuggingFace
mkdir logs

export EXPERIMENT_NAME=Run1
python3 -u compile.py 2>&1 | tee logs/$EXPERIMENT_NAME.logs
```

# Experiments

| Experiment Name| Expected Outcome | What's tested |
|------------|----------|---------------|
|Run1 | GPT-XL (Batch 5 / Seq 1024)| Compiled. `logs/Run1.compile`. Inference not accurate `logs/Run1.inference`|
|Run2 | GPT-XL (Batch 1/ Seq 1024) . Only changes the BATCH_SIZE from Run1 | Compiled. `logs/Run2.compile`. Inference not accurate `logs/Run2.inference` |
|Run3 | 10B Model. INIT_MODEL_FROM_CONFIG=1, BATCH_SIZE=1 |Compilation Failed - `logs/Run3.inference`|
|Run4 | GPT-XL (Batch 1 / Seq 64). BATCH_SIZE=1, MAX_SEQ_LENGTH=64 |Compiled. `logs/Run4.compile`. Inference not accurate `logs/Run4.inference`|
|Run5 | 6B Model INIT_MODEL_FROM_CONFIG=1, EMBEDDING=4096, LAYER=32, HEAD=32| Compilation Failed - `logs/Run5.inference`|
|Run6 | 6B Model INIT_MODEL_FROM_CONFIG=1, EMBEDDING=4096, LAYER=32, HEAD=32, BATCH_SIZE=1|Compilation Failed - `logs/Run6.inference`|

TODO : Check Int8
