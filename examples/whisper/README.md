# Whisper

This document shows how to build and run a [whisper model](https://github.com/openai/whisper/tree/main) in TensorRT-LLM on a single GPU.
## Overview

The TensorRT-LLM Whisper example code is located in [`examples/whisper`](./). There are three main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Whisper model.
 * [`run.py`](./run.py) to run the inference on a single wav file, or [a HuggingFace dataset](https://huggingface.co/datasets/librispeech_asr) [\(Librispeech test clean\)](https://www.openslr.org/12).
 * [`run_faster_whisper.py`](./run_faster_whisper.py) to do benchmark comparison with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper/tree/master).

## Support Matrix
  * FP16

## Usage

The TensorRT-LLM Whisper example code locates at [examples/whisper](./). It takes whisper pytorch weights as input, and builds the corresponding TensorRT engines.

### Build TensorRT engine(s)

Need to prepare the whisper checkpoint first by downloading models from [here](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L27-L28).


```bash
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
# tiny model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
```

TensorRT-LLM Whisper builds TensorRT engine(s) from the pytorch checkpoint.

```bash
# install requirements first
pip install -r requirements.txt

# Build the tiny model using a single GPU with plugins.
python3 build.py --output_dir tinyrt --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin
```

### Run

```bash
output_dir=./tinyrt
# decode a single audio file
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav
# decode a whole dataset
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_tiny_plugin
```

### Acknowledgment

This implementation of TensorRT-LLM for Whisper has been adapted from the [NVIDIA TensorRT-LLM Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) submission of Jinheng Wang, which can be found in the repository [Eddie-Wang-Hackathon2023](https://github.com/Eddie-Wang1120/Eddie-Wang-Hackathon2023) on GitHub. We extend our gratitude to Jinheng for providing a foundation for the implementation.
