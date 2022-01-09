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

Summary of Experiments & Opens we are trying to answer
* GPT-XL 1.5B Compiles; 
    * Does it produce valid output & not garbage ?
    * After changing SeqLength1024/Batch5 whats the speed up ?
    * Whats he GPU Usage & Model file size ?
* 6B Model Compilation fails ?
    * Does changing batch size 1 have an impact ? 
    * Whats the impact of WS_GB changes ?
    * How can a model of size X fail with Y memory on GPU ?
* 10B Model. 
    * Does this work with Int8 ?
* Whats the largest model we can compile & host on 40GB GPU Memory.


| Experiment Name| Expected Outcome | What's tested |
|------------|----------|---------------|
|Run1 | GPT-XL (Batch 5 / Seq 1024)| We expect this to compile. But have issues with inference. Garbage. If this works this is a more representative 1 for running benchmark tests.|
|Run2 | GPT-XL (Batch 1/ Seq 1024) . Only changes the BATCH_SIZE from Run1 | Compare this with Run1 for GPU Usage |
|Run3 | 10B Model. INIT_MODEL_FROM_CONFIG=1, BATCH_SIZE=1 |10GB Model. We expect compilation to fail|
|Run4 | GPT-XL (Batch 1 / Seq 64). BATCH_SIZE=1, MAX_SEQ_LENGTH=64 |Has worked before. The claimed 6x speedup is from this config.|Check if sequence length has an impact on model size|
|Run5 | 6B Model INIT_MODEL_FROM_CONFIG=1, EMBEDDING=4096, LAYER=32, HEAD=32| Compilation should fail |
|Run6 | 6B Model INIT_MODEL_FROM_CONFIG=1, EMBEDDING=4096, LAYER=32, HEAD=32, BATCH_SIZE=1|Is there is any difference between BatchSize 5 & 1 for Compilation Failure|
|RunX | Something in Int8.| Does GPU Usage change ?|
