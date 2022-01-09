FROM tensorrt-ubuntu20.04-cuda11.4:latest

USER root
RUN apt-get update
RUN apt-get install -y vim


RUN pip install nvidia-pyindex
RUN pip install polygraphy
RUN pip install jupyterlab 
RUN pip install torch==1.9.1+cu111 \
    	torchvision==0.10.1+cu111 \
    	torchaudio===0.9.1 \
        transformers \
    	-f https://download.pytorch.org/whl/torch_stable.html

RUN pip install tabulate toml sentencepiece onnx_graphsurgeon polygraphy --extra-index-url https://pypi.ngc.nvidia.com

