# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os 
import json
import argparse
import ast
import csv
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)
from reader import FileIterableDataset, RotatingFileWriter, MultiProcessFileWriter
from inflight import InFlighter
from transformers import AutoTokenizer
import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

def str2bool(value):
    """str2bool"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument('--enable_trie',
                        type=int,
                        default=0,
                        help='enable trie search.')
    parser.add_argument('--do_basic_tokenize',
                        type=int,
                        default=1,
                        help='enable basic_tokenize.')
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--second_engine_dir', type=str, default=None)
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)

    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)

    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)
    parser.add_argument('--output_cum_log_probs',
                        type=int,
                        help='cum_log_probs need stored',
                        default=None)

    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='zeus')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers",
                        default='zeus/vocab.txt')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--early_stopping',
                        type=int,
                        help='Use early stopping if num_beams > 1'
                        '1 for early-stopping, 0 for non-early-stopping'
                        'other values for stopping by length',
                        default=1)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tokenize_chinese_chars', default=True, type=str2bool)
    parser.add_argument('--drop_last',
        default=False,
        action='store_true',
        help="drop last examples.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help="predict ouput directory.")
    parser.add_argument('--inflight_batching',
                        default=False,
                        action='store_true',
                        help="inflight mode for faster inference.")
    parser.add_argument('--cuda_graph_mode',
                        default=False,
                        action='store_true',
                        help="cuda graph mode.")
    parser.add_argument('--decoder_per_request',
                        default=False,
                        action='store_true',
                        help="decoder per request.")
    parser.add_argument('--dataloader_worker_num', type=int, default=2)
    parser.add_argument('--sort_example',
                        default=True,
                        type=str2bool,
                        help="sort example.")
    parser.add_argument('--output_worker_num', type=int, default=2)
    parser.add_argument('--customize_position_id',
                        default=False,
                        action='store_true',
                        help="can customize the position id.")
    parser.add_argument('--input_id_format', type=int, default=0)
    parser.add_argument('--max_concurrent_tasks',
                        type=int,
                        help="max concurrent tasks",
                        default=1024)
    parser.add_argument('--output_costtime', 
                        default=False,
                        action='store_true',
                        help="output cost time.")
    parser.add_argument('--output_avglength', 
                        default=False,
                        action='store_true',
                        help="output avg length.")

    return parser.parse_args(args=args)

def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 cum_log_probs=None,
                 log_probs=None,
                 output_logits_npy=None,
                 output_cum_log_probs_npy=None,
                 output_log_probs_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(
                    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    # Save context logits
    if context_logits is not None and output_logits_npy is not None:
        context_logits = torch.cat(context_logits, axis=0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])

        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)

    # Save generation logits
    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)

    # Save cum log probs
    if cum_log_probs is not None and output_cum_log_probs_npy is not None:
        cum_log_probs_file = Path(output_cum_log_probs_npy)
        cum_log_probs_outputs = np.array(cum_log_probs.cpu().contiguous(),
                                         dtype='float32')
        np.save(cum_log_probs_file, cum_log_probs_outputs)

    # Save cum log probs
    if log_probs is not None and output_log_probs_npy is not None:
        log_probs_file = Path(output_log_probs_npy)
        log_probs_outputs = np.array(log_probs.cpu().contiguous(),
                                     dtype='float32')
        np.save(log_probs_file, log_probs_outputs)


def run_inflight(engine_dir,
                 tokenizer,
                 dataloader,
                 file_writer,
                 args):
    """ run in flight """
    additional_options = ['batch_size', 'num_beams', 'temperature', 'top_k', 'top_p',
            'length_penalty', 'repetition_penalty', 'presence_penalty', 'beam_search_diversity_rate',
            'frequency_penalty', 'early_stopping', 'max_concurrent_tasks', 'max_output_len']
    options = {k: getattr(args, k) for k in additional_options}
    options['update_interval'] = args.batch_size
    flighter = InFlighter(engine_dir,
                          tokenizer,
                          dataloader,
                          file_writer.enqueue,
                          **options)
    begin_time = time.time()
    flighter.run()
    flighter.close()
    total_time_cost = max(0.01, time.time() - begin_time)
    file_writer.close()
    print(f'run finished. total number: {flighter.example_num}'
          f', total cost: {total_time_cost:.2f} s'
          f', speed: {flighter.example_num / total_time_cost:.2f} samples/sec')

def get_tokenizer_config(model_name_or_path):
    """ Load tokenizer configuration from local files """
    result = {}
    config_file = os.path.join(model_name_or_path, "tokenizer_config.json")
    with open(config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    return result

def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)
    print("model_version=", model_version)
    if model_version == "glm":
        # zeus tokenizer load
        from zeus.tokenization_zeus import ZeusTokenizer
        tokenizer = ZeusTokenizer.from_pretrained(args.vocab_file,
                        tokenize_chinese_chars=args.tokenize_chinese_chars)
    else:
        # qwen or other tokenizer load
        tokenizer_config = get_tokenizer_config(args.tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, config=tokenizer_config, trust_remote_code=True)
    tokenizer.do_basic_tokenize = bool(args.do_basic_tokenize)
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id
    print("use do_basic_tokenize:", args.do_basic_tokenize, ", pad_id: ", pad_id, ", end_id: ", end_id)

    dataset = FileIterableDataset(input_file=args.input_file,
                                tokenizer=tokenizer,
                                batch_size=args.batch_size,
                                drop_last=args.drop_last,
                                add_special_tokens=args.add_special_tokens,
                                max_input_length=args.max_input_length,
                                input_id_format=args.input_id_format,
                                sort_example=args.sort_example)
    dataloader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=True,
                            num_workers=args.dataloader_worker_num)
    if args.inflight_batching:
        decode_func = dataset.decode_output_v2
    else:
        decode_func = dataset.decode_output
    file_writer = MultiProcessFileWriter(args.output_dir,
                                         decode_func,
                                         worker_num=args.output_worker_num)

    if args.inflight_batching:
        print('run in flight ...')
        return run_inflight(args.engine_dir,
                            tokenizer,
                            dataset,
                            file_writer,
                            args)

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         lora_ckpt_source=args.lora_ckpt_source)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        #assert args.use_py_session, "Medusa is only supported by py_session"
        #assert args.temperature == 0, "Medusa should use temperature == 0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=None,#len(batch_input_ids),
            max_input_len=None,#max(input_lengths),
            max_output_len=args.max_output_len,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            inflight_batching=args.inflight_batching,
            cuda_graph_mode=args.cuda_graph_mode,
            decoder_per_request=args.decoder_per_request,
            enable_trie=args.enable_trie,
            end_id=end_id,
        )
    runner = runner_cls.from_dir(**runner_kwargs)
    if args.second_engine_dir is not None:
        runner.add_engine(args.second_engine_dir)
    
    if args.customize_position_id:
        from zeus.pos_id_manager import PosIdManager
        pos_id_manager = PosIdManager()
        pos_id_manager.init(args.max_input_length + args.max_output_len, "GLM")
    
    def generator():
        for _ in dataloader:
            yield _

    output_cum_log_probs = ((args.output_cum_log_probs_npy is not None) \
                            and (args.output_cum_log_probs is not None))
    Tqdm = tqdm(ncols=50)
    example_num = 0
    gtime_cost = 0
    begin_time = time.time()
    infer_time = []
    for batch_dict in generator():
        qids = batch_dict['qids']
        batch_input_ids = batch_dict['input_ids']
        input_lengths = batch_dict['input_lengths']

        with torch.no_grad():
            t = time.time()
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=args.max_output_len,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                beam_search_diversity_rate=args.beam_search_diversity_rate,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                enable_trie=args.enable_trie,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                output_cum_log_probs=output_cum_log_probs,
                output_log_probs=(args.output_log_probs_npy != None),
                lora_uids=args.lora_task_uids,
                prompt_table_path=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                streaming=args.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=args.medusa_choices)
            torch.cuda.synchronize()
            gtime_cost += (time.time() - t)
            example_num += len(qids)
            span_time = time.time() - t
            infer_time.append(span_time)

        if args.streaming:
            for curr_outputs in throttle_generator(outputs,
                                                args.streaming_interval):
                if runtime_rank == 0:
                    output_ids = curr_outputs['output_ids']
                    sequence_lengths = curr_outputs['sequence_lengths']
                    cum_log_probs = None
                    log_probs = None
                    if output_cum_log_probs:
                        cum_log_probs = outputs['cum_log_probs']
                    if args.output_log_probs_npy != None:
                        log_probs = outputs['log_probs']
                    print_output(
                        tokenizer,
                        output_ids,
                        input_lengths,
                        sequence_lengths,
                        output_csv=args.output_csv,
                        output_npy=args.output_npy,
                        cum_log_probs=cum_log_probs,
                        log_probs=log_probs,
                        output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                        output_log_probs_npy=args.output_log_probs_npy)
        else:
            if runtime_rank == 0:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                cum_log_probs = None
                log_probs = None
                nlu_scores = None
                context_features = None
                if runner.gather_context_logits:
                    context_logits = outputs['context_logits']
                if runner.gather_generation_logits:
                    generation_logits = outputs['generation_logits']
                if output_cum_log_probs:
                    cum_log_probs = outputs['cum_log_probs']
                    cum_log_probs = cum_log_probs.tolist()
                if args.output_log_probs_npy != None:
                    log_probs = outputs['log_probs']
                if 'nlu_scores' in outputs:
                    nlu_scores = outputs['nlu_scores']
                    nlu_scores = nlu_scores.tolist()
                if outputs.get('context_features') is not None:
                    context_features = outputs['context_features'].tolist()
                batch_size, beam_size, _ = output_ids.size()
                file_writer.enqueue(
                                output_ids.tolist(),
                                batch_size,
                                beam_size,
                                qids,
                                input_lengths,
                                sequence_lengths.tolist(),
                                cum_log_probs,
                                nlu_scores,
                                context_features,
                                span_time if args.output_costtime else None)
                # print_output(tokenizer,
                #             output_ids,
                #             input_lengths,
                #             sequence_lengths,
                #             output_csv=args.output_csv,
                #             output_npy=args.output_npy,
                #             context_logits=context_logits,
                #             generation_logits=generation_logits,
                #             output_logits_npy=args.output_logits_npy,
                #             cum_log_probs=cum_log_probs,
                #             log_probs=log_probs,
                #             output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                #             output_log_probs_npy=args.output_log_probs_npy)
        Tqdm.update(args.batch_size)
    Tqdm.close()
    total_time_cost = max(0.01, time.time() - begin_time)
    file_writer.close()
    print(f'run finished. total number: {example_num}'
          f', total cost: {total_time_cost:.2f} s'
          f', speed: {example_num / total_time_cost:.2f} samples/sec'
          f', generate cost: {gtime_cost:.2f} s'
          f', speed: {example_num / gtime_cost:.2f} samples/sec.')

    if len(infer_time) > 0:
        print(f"infer time per value, "
        f"25%: {np.percentile(infer_time, 25):.6f}, 50%: {np.percentile(infer_time, 50):.6f}, "
        f"75%: {np.percentile(infer_time, 75):.6f}, 95%: {np.percentile(infer_time, 95):.6f}, "
        f"97%: {np.percentile(infer_time, 97):.6f}, 99%: {np.percentile(infer_time, 99):.6f}")

    if args.output_avglength:
        file_writer.print_avg_len()

    if args.run_profiling:
        ite = 10
        # warmup
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping,
                    enable_trie=args.enable_trie,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    lora_uids=args.lora_task_uids,
                    prompt_table_path=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()

        tensorrt_llm.profiler.start("tmp")
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping,
                    enable_trie=args.enable_trie,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    lora_uids=args.lora_task_uids,
                    prompt_table_path=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()
        tensorrt_llm.profiler.stop("tmp")

        print(
            f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

