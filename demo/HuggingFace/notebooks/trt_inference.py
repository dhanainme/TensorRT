import sys
import os

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    GPT2TokenizerFast
)

from GPT2.trt import GPT2TRTDecoder, GPT2TRTEngine
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.networks import NetworkMetadata, Precision



GPT_VARIANT = 'gpt2-large'
BATCH_SIZE=1
PRECISION='fp16'
TRT_ENGINE='models/gpt2/trt-engine/gpt2.onnx.engine'
MAX_TOKEN_LENGTH=64

metadata=NetworkMetadata(GPT_VARIANT, Precision(PRECISION), None)
trt_gpt_config = GPT2Config(GPT_VARIANT)
trt_engine = GPT2TRTEngine(TRT_ENGINE,metadata,batch_size=BATCH_SIZE)
trt_decoder = GPT2TRTDecoder(trt_engine, metadata, trt_gpt_config, batch_size=BATCH_SIZE)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

input_text = "HBO released a new TV series this weekend."
inputs = tokenizer([input_text] * BATCH_SIZE, return_tensors="pt")


print("===========================================")
print(f"Input text / input tensors / input shape : ", input_text , inputs, inputs['input_ids'].shape)

output_tensor = trt_decoder.generate(inputs['input_ids'].cuda(), max_length=MAX_TOKEN_LENGTH)
output_text = tokenizer.decode(output_tensor[0], skip_special_tokens=True)
print(f"==========================================\nTRT Output text / output tensors \n",output_text,'\n',output_tensor)
