# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os

# isort: off
import torch
import tensorrt as trt
# isort: on

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.models import BertForQuestionAnswering, BertModel
from tensorrt_llm.runtime import Session, TensorInfo
from transformers import BertTokenizer

tensorrt_llm.logger.set_level('verbose')

from build import get_engine_name  # isort:skip

enc = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking', trust_remote=True)

mask_token_id = enc.convert_tokens_to_ids('[MASK]')
logger.info(f"Mask token ID: {mask_token_id}")

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, default='bert_outputs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    model_name = config['builder_config']['name']
    runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = get_engine_name(model_name, dtype, world_size,
                                     runtime_rank)
    serialize_path = os.path.join(args.engine_dir, serialize_path)

    stream = torch.cuda.current_stream().cuda_stream
    logger.info(f'Loading engine from {serialize_path}')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine')
    session = Session.from_serialized_engine(engine_buffer)
    
    
    text = "Replace me by any text you'd like."
    encoded_input = enc(text, return_tensors='pt')
    logger.info(f'enconded_input: {encoded_input}')
    logger.info(f'type of encoded_input: {type(encoded_input)}')
    logger.info(f'encoded_input.keys(): {encoded_input.keys()}')

    masked_sentences = ['Paris is the [MASK] of France.',
                        'The primary [MASK] of the United States is English.',
                        'A baseball game consists of at least nine [MASK].',
                        'Topology is a branch of [MASK] concerned with the properties of geometric objects that remain unchanged under continuous transformations.']
    pos_masks = [4, 3, 9, 6]

    # Tokenize the input
    encoded_inputs = enc(masked_sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)

    input_ids = encoded_inputs['input_ids'].cuda()
    attention_masks = encoded_inputs['attention_mask'].cuda()
    # Token type IDs are not necessary for MLM but required if your model uses them
    token_type_ids = torch.zeros_like(input_ids).cuda()  # Assuming a single segment input for simplicity

    # Calculate the actual lengths of the input for custom handling, if necessary
    input_lengths = attention_masks.sum(dim=1)

    inputs = {
        'input_ids': input_ids,
        'input_lengths': input_lengths,  
        'token_type_ids': token_type_ids
    }

    # Presumably, your session.infer_shapes function prepares for dynamic shape inference
    # This step might need adjustments based on your specific implementation details
    output_info = session.infer_shapes([
        TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
        TensorInfo('input_lengths', trt.DataType.INT32, input_lengths.shape),
        TensorInfo('token_type_ids', trt.DataType.INT32, token_type_ids.shape),
    ])
    
    outputs = {
        t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device='cuda') for t in output_info
    }

    # Determine the correct output_name based on your model's configuration
    if (model_name == BertModel.__name__):
        output_name = 'hidden_states'
    elif (model_name == BertForQuestionAnswering.__name__):
        output_name = 'logits'
    else:
        assert False, f"Unknown BERT model {model_name}"

    # Run inference
    ok = session.run(inputs, outputs, stream)
    if not ok:
        raise RuntimeError("Runtime execution failed")

    # Wait for all operations on the GPU to complete
    torch.cuda.synchronize()

    # Process the output to find the most likely token IDs for each mask
    logits = outputs[output_name]  # Assuming this is where your logits are
    logger.info(f"Logits shape: {logits.shape}")

    for i, pos_mask in enumerate(pos_masks):
        logits_at_mask = logits[i, pos_mask, :]
        probs = torch.softmax(logits_at_mask, dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)
        print(f"Top 5 predictions for mask {i}: {enc.convert_ids_to_tokens(top_ids.cpu().numpy())} with probs {top_probs.cpu().numpy()}")