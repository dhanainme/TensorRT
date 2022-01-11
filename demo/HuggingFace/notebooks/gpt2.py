#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Accelerating HuggingFace GPT-2 Inference with TensorRT
# 
# GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. The model was pretrained on the raw texts to guess the next word in sentences. As no human labeling was required, GPT-2 pretraining can use lots of publicly available data with an automatic process to generate inputs and labels from those data.
# 
# This notebook shows 3 easy steps to convert a [HuggingFace PyTorch GPT-2 model](https://huggingface.co/gpt2) to a TensorRT engine for high-performance inference.
# 
# 1. [Download HuggingFace GPT-2 model ](#1)
# 1. [Convert to ONNX format](#2)
# 1. [Convert to TensorRT engine](#3)
# 
# ## Prerequisite
# 
# Follow the instruction at https://github.com/NVIDIA/TensorRT to build the TensorRT-OSS docker container required to run this notebook.
# 
# Next, we install some extra dependencies and restart the Jupyter kernel.

# In[ ]:


#os.run_cell_magic('capture', '', '!pip3 install -r ../requirements.txt\n\nimport IPython\napp = IPython.Application.instance()\napp.kernel.do_shutdown(True)  ')


# In[ ]:


import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

import torch 

# huggingface
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
)


# <a id="1"></a>
# 
# ## 1. Download HuggingFace GPT-2 model 
# 
# First, we download the original HuggingFace PyTorch GPT-2 model from HuggingFace model hubs, together with its associated tokernizer.
# 
# The GPT-2 variants supported by TensorRT 8 are: gpt2 (117M), gpt2-large (774M).

# In[ ]:


# download model and tokernizer
GPT2_VARIANT = 'gpt2' # choices: gpt2 | gpt2-large

model = GPT2LMHeadModel.from_pretrained(GPT2_VARIANT)

config = GPT2Config(GPT2_VARIANT)
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_VARIANT)


# In[ ]:


# save model locally
pytorch_model_dir = './models/{}/pytorch'.format(GPT2_VARIANT)
os.system('mkdir -p $pytorch_model_dir')

model.save_pretrained(pytorch_model_dir)
print("Pytorch Model saved to {}".format(pytorch_model_dir))


# ### Inference with PyTorch model
# 
# #### Single example inference

# In[ ]:


# carry out inference with a single sample
inputs = tokenizer("Hello, my dog is ", return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])

logits = outputs.logits


# For benchmarking purposes, we will employ a helper function `gpt2_inference` which executes the inference on a single batch repeatedly and measures end to end execution time. Let's take note of this execution time for later comparison with TensorRT. 
#  
# `TimingProfile` is a named tuple that specifies the number of experiments and number of times to call the function per iteration (and number of warm-up calls although it is not used here).

# In[ ]:


from GPT2.measurements import gpt2_inference
from NNDF.networks import TimingProfile

# Benchmarking TensorRT performance on single batch
output, decoder_e2e_median_time = gpt2_inference(
            model.to('cuda:0'), inputs.input_ids.to('cuda:0'), TimingProfile(iterations=10, number=1, warmup=1)
        )
decoder_e2e_median_time


# #### Open-end text generation
# Next, we will employ the PyTorch model for the open-end text generation task, which GPT-2 is particularly good at. 

# In[ ]:


from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig

sample_output = model.to('cuda:0').generate(inputs.input_ids.to('cuda:0'), max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH['gpt2'])

# de-tokenize model output to raw text
tokenizer.decode(sample_output[0], skip_special_tokens=True)


# For benchmarking purposes, we will employ a helper function `full_inference_greedy` which executes the inference repeatedly and measures end to end execution time. Let's take note of this execution time for later comparison with TensorRT. 
#  
# TimingProfile is a named tuple that specifies the number of experiments and number of times to call the function per iteration (and number of warm-up calls although it is not used here).

# In[ ]:


from GPT2.measurements import full_inference_greedy

# get complete decoder inference result and its timing profile
sample_output, full_e2e_median_runtime = full_inference_greedy(
    model.to('cuda:0'), inputs.input_ids, TimingProfile(iterations=10, number=1, warmup=1),
    max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[GPT2_VARIANT]
)
full_e2e_median_runtime


# <a id="2"></a>
# 
# ## 2. Convert to ONNX format
# 
# Prior to converting the model to a TensorRT engine, we will first convert the PyTorch model to an intermediate universal format: ONNX.
# 
# ONNX is an open format for machine learning and deep learning models. It allows you to convert deep learning and machine learning models from different frameworks such as TensorFlow, PyTorch, MATLAB, Caffe, and Keras to a single format.
# 
# At a high level, the steps to convert a PyTorch model to TensorRT are as follows:
# - Convert the pretrained image segmentation PyTorch model into ONNX.
# - Import the ONNX model into TensorRT.
# - Apply optimizations and generate an engine.
# - Perform inference on the GPU with the TensorRT engine. 

# In[ ]:


from GPT2.export import GPT2TorchFile
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.networks import NetworkMetadata, Precision


# In[ ]:


metadata=NetworkMetadata(GPT2_VARIANT, Precision('fp16'), None)
gpt2 = GPT2TorchFile(model.to('cpu'), metadata)


# In[ ]:


os.system('mkdir -p ./models/$GPT2_VARIANT/ONNX')

onnx_path = ('./models/{}/ONNX/model.onnx'.format(GPT2_VARIANT))
gpt2.as_onnx_model(onnx_path, force_overwrite=False)

del model


# <a id="3"></a>
# 
# ## 3. Convert to TensorRT engine
# 
# Now we are ready to parse the ONNX model and convert it to an optimized TensorRT model.
# 
# Note: As TensorRT carries out many optimization, this conversion process for the larger model might take a while.

# In[ ]:


from GPT2.export import GPT2ONNXFile


# In[ ]:


os.system('mkdir -p ./models/$GPT2_VARIANT/trt-engine')
trt_path = './models/{}/trt-engine/{}.onnx.engine'.format(GPT2_VARIANT, GPT2_VARIANT)
gpt2_engine = GPT2ONNXFile(onnx_path, metadata).as_trt_engine(trt_path)


# In[ ]:


gpt2_engine.fpath


# ### Inference with TensorRT engine
# 
# Great, if you have reached this stage, it means we now have an optimized TensorRT engine for the GPT-2 model, ready for us to carry out inference. 
# 
# The GPT-2 model with TensorRT backend can now be employed in place of the original HuggingFace GPT-2 model.

# #### Single batch inference
# 

# In[ ]:


from GPT2.trt import GPT2TRTDecoder

gpt2_trt = GPT2TRTDecoder(gpt2_engine, metadata, config)

outputs = gpt2_trt(inputs.input_ids)
logits = outputs.logits


# In[ ]:


# Benchmarking TensorRT performance on single batch
output, decoder_e2e_median_time = gpt2_inference(
            gpt2_trt, inputs.input_ids, TimingProfile(iterations=10, number=1, warmup=1)
        )
decoder_e2e_median_time


# #### Open-end text generation

# In[ ]:


sample_output = gpt2_trt.generate(inputs.input_ids.to('cuda:0'), max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH['gpt2'])

# de-tokenize model output to raw text
tokenizer.decode(sample_output[0], skip_special_tokens=True)


# In[ ]:


# get complete decoder inference result and its timing profile
sample_output, full_e2e_median_runtime = full_inference_greedy(
    gpt2_trt, inputs.input_ids, TimingProfile(iterations=10, number=1, warmup=1),
    max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH['gpt2']
)
full_e2e_median_runtime


# You can now compare the output of the original PyTorch model and the TensorRT engine. Notice the speed difference. On an NVIDIA V100 32GB GPU, this results in about ~5x performance improvement for the GPT-2 small model (from an average of 0.704s to 0.134s).

# ## Conclusion and where-to next?
# 
# This notebook has walked you through the process of converting a HuggingFace PyTorch GPT-2 model to an optimized TensorRT engine for inference in 3 easy steps. The TensorRT inference engine can be conviniently used as a drop-in replacement for the orginial HuggingFace GPT-2 model while providing significant speed up. 
# 
# If you are interested in further details of the conversion process, check out [GPT2/trt.py](../GPT2/trt.py)

# In[ ]:




