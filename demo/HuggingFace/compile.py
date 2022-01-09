import json
import sys
import os
from time import sleep
from pprint import pprint
import torch

from GPT2.export import GPT2TorchFile, GPT2ONNXFile
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.networks import NetworkMetadata, Precision
from GPT2.trt import GPT2Config, GPT2Tokenizer, GPT2TRTDecoder
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config, GPT2TokenizerFast
import numpy as np
import time


MAX_SEQ_LENGTH = int(os.getenv("MAX__SEQ_LENGTH", "1024"))
GPT_VARIANT=os.getenv("GPT_VARIANT", "gpt2-xl")
VOCAB_LENGTH=int(os.getenv("VOCAB_LENGTH", "50257"))
BATCH_SIZE=int(os.getenv("BATCH_SIZE", "5"))
PRECISION=os.getenv("PRECISION", "fp16")
WS_GB=int(os.getenv("WS_GB", "30"))

EXPERIMENT_NAME=os.getenv("EXPERIMENT_NAME", "gpt_xl")
EXPORT_PATH=os.getenv("EXPORT_PATH", f"/workspace/export/{EXPERIMENT_NAME}/")

INIT_MODEL_FROM_CONFIG=int(os.getenv("INIT_MODEL_FROM_CONFIG", "0"))
EMBEDDING=int(os.getenv("EMBEDDING", "3588"))
LAYERS=int(os.getenv("LAYER","80"))
HEADS=int(os.getenv("HEAD", "52"))
MAX_TOKEN_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH", "48"))
MAX_OUT_LENGTH = int(os.getenv("MAX_OUT_LENGTH", "512"))


def print_globs():
    print("===============================================================================")
    VARS = ["MAX_SEQ_LENGTH", "GPT_VARIANT", "VOCAB_LENGTH", "BATCH_SIZE", "PRECISION", "WS_GB"]
    VARS += ["EXPERIMENT_NAME", "EXPORT_PATH", "EMBEDDING", "LAYERS", "HEADS", "MAX_TOKEN_LENGTH", "MAX_OUT_LENGTH"]
    for var in VARS:
        print(var," : ",globals()[var])
    print("===============================================================================")


def get_model_and_tokenizer():
    
    model = None
    if(INIT_MODEL_FROM_CONFIG == 1): 
        configuration = GPT2Config(n_embd=EMBEDDING, n_layer=LAYERS, n_head=HEADS)
        print(f"Creating model from Config n_embd/n_layer/n_head : {EMBEDDING}/{LAYERS}/{HEADS}. Ignoring GPT_VARIANT")
        model = GPT2LMHeadModel(configuration)
    else:
        print(f"Creating model from variant name {GPT_VARIANT}, Ignoring EMBEDDING, LAYERS, HEADS")
        model = GPT2LMHeadModel.from_pretrained(GPT_VARIANT)

    print(f"Model created")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if(PRECISION=="fp16"):
        model = model.half()

    def get_parameter_count(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"Params {get_parameter_count(model)}")
    cuda_model = model.cuda()
    return cuda_model, tokenizer


def compile_onnx_model(torch_model, onnx_path):
    print(f'Compiling to ONNX Model - {onnx_path}')
    metadata=NetworkMetadata(GPT_VARIANT, Precision(PRECISION), None)
    gpt2 = GPT2TorchFile(torch_model, metadata)
    onnx_model = gpt2.as_onnx_model(onnx_path, force_overwrite=False)
    return onnx_model


def compile_trt_model(onnx_path, trt_path):
    print(f'Compiling to TRT Model - {trt_path}')
    metadata=NetworkMetadata(GPT_VARIANT, Precision(PRECISION), None)
    gpt2_engine = GPT2ONNXFile(onnx_path, metadata).as_trt_engine(trt_path, batch_size=BATCH_SIZE)
    return gpt2_engine



def prep_paths(export_dir):
    import shutil
    shutil.rmtree(export_dir,ignore_errors=True)
    os.makedirs(f'{export_dir}/ONNX/')
    os.makedirs(f'{export_dir}/TRT/')
    return f'{export_dir}/ONNX/model.onnx', f'{export_dir}/TRT/model.engine'

def main():

    print_globs()
    torch_model, tokenizer = get_model_and_tokenizer()

    print("===========================================================")
    print("Model configuration")
    pprint(torch_model.config)
    print("===========================================================")

    torch_model.resize_token_embeddings(len(tokenizer))
    os.system("nvidia-smi")
    onnx_path, trt_path = prep_paths(export_dir=EXPORT_PATH)
    onnx_model = compile_onnx_model(torch_model, onnx_path)

    os.system("nvidia-smi")
    del torch_model
    del onnx_model
    del tokenizer
    torch.cuda.empty_cache()

    os.system("nvidia-smi")
    trt_engine = compile_trt_model(onnx_path, trt_path)

if __name__ == "__main__":
    main()
