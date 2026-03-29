""" data reader
"""

import os
import csv
import json
import time
import heapq
import torch
import multiprocessing
import numpy as np
from collections import namedtuple
from tensorrt_llm import logger
from torch.utils.data import IterableDataset, DataLoader


class FileIterableDataset(IterableDataset):
    def __init__(self,
                input_file,
                tokenizer,
                batch_size,
                drop_last,
                add_special_tokens,
                max_input_length,
                remove_input_padding=False,
                input_id_format=0,
                sort_example=False,
                return_pt=True,
                chat_template=False):
        super(FileIterableDataset).__init__()

        if os.path.isfile(input_file):
            self.input_files = [input_file]
        elif os.path.isdir(input_file):
            files = os.listdir(input_file)
            self.input_files = [os.path.join(input_file, f) for f in files]
        else:
            raise FileNotFoundError(input_file)

        self.is_json_file = self.input_files[0].endswith('.json')
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_special_tokens = add_special_tokens
        self.max_input_length = max_input_length
        self.remove_input_padding = remove_input_padding
        self.step = 0
        self.input_id_format = input_id_format
        self.sort_example = sort_example
        self.return_pt = return_pt
        self.chat_template = chat_template

    def __iter__(self):
        for batched in self.batched_iter(self.file_iter(self.input_files)):
            yield self.prepare_inputs(batched)

    def batched_iter(self, examples):
        """ batched iter """
        batched = []
        worker_info = torch.utils.data.get_worker_info()
        for index, example in enumerate(examples):
            # support multiprocess dataloader
            if worker_info and index % worker_info.num_workers != worker_info.id:
                continue
            batched.append(example)
            if len(batched) == self.batch_size:
                yield batched
                batched = []
        
        if not self.drop_last and len(batched) > 0:
            yield batched
            batched = []


    def file_iter(self, file_list, prefactor=50):
        """ file_iter """
        prefactor = self.batch_size * prefactor

        i = 0
        line_heap = []
        if self.is_json_file:
            self.Example = namedtuple("Example", ["qid", "src"])
            for file_path in file_list:
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        try:
                            i += 1
                            line = line.strip()
                            obj = json.loads(line)
                            example = self.Example(obj['qid'], obj['prompt'])
                            heapq.heappush(line_heap,
                                        (len(example.src) if self.sort_example else i, example))
                            if len(line_heap) < prefactor:
                                continue
                            yield heapq.heappop(line_heap)[1]
                        except Exception as e:
                            logger.error(f'Reading Error: filename[{file_path}] linenum[{i}] data{line} error[{e}]')
        else:
            for file_path in file_list:
                with open(file_path, 'r') as f:
                    reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    try:
                        self.Example = namedtuple("Example", next(reader))
                    except StopIteration:
                        logger.warning(f'Empty file: {file_path}')
                        continue
                    for line in reader:
                        try:
                            i += 1
                            example = self.Example(*line)
                            heapq.heappush(line_heap,
                                        (len(example.src) if self.sort_example else i, example))
                            if len(line_heap) < prefactor:
                                continue
                            yield heapq.heappop(line_heap)[1]
                        except Exception as e:
                            logger.error(f'Reading Error: filename[{file_path}] linenum[{i}] data{line} error[{e}]')

        while len(line_heap) > 0:
            yield heapq.heappop(line_heap)[1]

    def prepare_inputs(self, batched):
        """ prepare inputs """
        batched_qids = []
        batched_input_ids = []
        batched_input_lengths = []
        for example in batched:
            if self.chat_template:
                prompt = example.src
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
                input_ids = self.tokenizer.encode(
                        text,
                        add_special_tokens=self.add_special_tokens,
                        truncation=True,
                        max_length=self.max_input_length - 1  # bos_token_id
                )
            else:
                input_ids = self.tokenizer.encode(
                    example.src,
                    add_special_tokens=self.add_special_tokens,
                    truncation=True,
                    max_length=self.max_input_length - 1  # bos_token_id
                )

            if self.input_id_format == 0:
                # [token1] [token2] ... [gMASK] [START]
                input_ids.append(self.tokenizer.bos_token_id)
            elif self.input_id_format == 1:
                # [START] [token1] [token2] ...
                input_ids.insert(0, self.tokenizer.bos_token_id)
                input_ids = input_ids[:-1]
            # else:
            #     raise ValueError('Invalid input_id_format value: {}'.format(self.input_id_format))

            batched_qids.append(example.qid)
            if self.return_pt:
                batched_input_ids.append(torch.tensor(input_ids, dtype=torch.int32))
            else:
                batched_input_ids.append(input_ids)
            batched_input_lengths.append(len(input_ids))

        return {
            'qids': batched_qids,
            'input_ids': batched_input_ids,
            'input_lengths': batched_input_lengths
        }

    def _decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

    def decode_output(self,
                      output_ids,
                      batch_size,
                      num_beams,
                      qids,
                      input_lengths,
                      sequence_lengths,
                      cum_log_probs,
                      nlu_scores,
                      context_features,
                      cost_time):
        """ decode output """
        output_list = []
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]]
            tmp_dict = {
                'qid': qids[batch_idx],
                'src': self._decode(inputs),
                'tgt': []
            }
            if cost_time is not None:
                tmp_dict["cost"] = "%sMS" % (cost_time * 1000)
            if context_features is not None:
                tmp_dict['context_features'] = context_features[batch_idx]
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end]
                tgt = self._decode(outputs)
                if cum_log_probs is not None:
                    tgt = tgt + ' Probs<%.7f>' % cum_log_probs[batch_idx][beam]
                if nlu_scores is not None:
                    scores = [f"{x:.7f}" for x in nlu_scores[batch_idx * num_beams + beam]]
                    tgt += ' Scores<' + ' '.join(scores) + '>'
                tmp_dict['tgt'].append(tgt)

            if self.step < 5:
                print(json.dumps(tmp_dict, indent=2, ensure_ascii=False))
                self.step += 1
            output_list.append(tmp_dict)
            
        return output_list

    def decode_output_v2(self,
                         qid,
                         src_token_ids,
                         tgt_token_ids,
                         cum_log_probs=None):
        """ decode output v2 """
        tmp_dict = {
            'qid': qid,
            'src': self._decode(src_token_ids),
            'tgt': []
        }
        for idx in range(len(tgt_token_ids)):
            tgt = self._decode(tgt_token_ids[idx])
            if cum_log_probs is not None:
                tgt = tgt + ' Probs<%f>' % cum_log_probs[idx]
            tmp_dict['tgt'].append(tgt)

        if self.step < 5:
            print(json.dumps(tmp_dict, indent=2, ensure_ascii=False))
            self.step += 1

        return tmp_dict


class RotatingFileWriter(object):
    """ RotatingFileWriter """
    part_index = 0

    def __init__(self, output_dir, max_file_len=5000, postfix='txt'):
        """ init """
        self.output_dir = output_dir
        self.max_file_len = max_file_len
        self.file_len = 0
        self.postfix = postfix

        self.file = None
        self.buff = []
        self.buff_len = min(5000, max_file_len)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_file(self):
        """ get file """
        if self.file_len >= self.max_file_len:
            if self.file:
                self.file.close()
                self.file = None

        if not self.file:
            filename = '{}/{:0>5d}.{}'.format(self.output_dir,
                                              __class__.part_index,
                                              self.postfix)
            self.file = open(filename, 'w')
            self.file_len = 0
            __class__.part_index += 1

        return self.file

    def flush(self):
        """ flush """
        if self.buff:
            self.get_file().write(''.join(self.buff))
            self.file_len += len(self.buff)
            self.buff = []

    def write(self, content):
        """ write """
        if not isinstance(content, list):
            content = [content]
        for x in content:
            if isinstance(x, dict):
                x = json.dumps(x, ensure_ascii=False) + '\n'
            self.buff.append(x)

        if len(self.buff) >= self.buff_len:
            self.flush()

    def close(self):
        """ close """
        self.flush()
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MultiProcessFileWriter(object):
    """ MultiProcessFileWriter """
    def __init__(self, output_dir, out_decoder, worker_num=5):
        self.output_dir = output_dir
        self.decoder = out_decoder
        self.worker_num = worker_num
        self.queue = multiprocessing.Queue()
        self.hasfailed = False

        def worker(idx, queue):
            """ worker """
            writer = RotatingFileWriter(output_dir, postfix=str(idx) + '.txt')
            while True:
                item = queue.get()
                if item is None:
                    writer.close()
                    logger.info(f'process worker-{idx} exit ...')
                    return
                writer.write(self.decoder(*item))

        self.workers = []
        for idx in range(worker_num):
            p = multiprocessing.Process(target=worker, args=(idx, self.queue))
            p.daemon = True
            p.start()
            self.workers.append(p)

    def enqueue(self, *args):
        """ enqueue """
        self.queue.put(args)

    def has_failed(self):
        """ has failed """
        workers = []
        for p in self.workers:
            if p.is_alive():
                workers.append(p)
            elif p.exitcode != 0:
                self.hasfailed = True
        self.workers = workers

        return self.hasfailed

    def close(self):
        """ close """
        for _ in range(self.worker_num):
            self.queue.put(None)
        while self.workers:
            self.has_failed()
            time.sleep(1)

        if self.hasfailed:
            # Prevent q.join_thread() from blocking
            self.queue.cancel_join_thread()
            return False
        else:
            return True

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if not self.close():
            raise Exception('sub process exitcode !=0 .')
    
    def print_avg_len(self):
        """ print avg len """
        line_cnt = 0
        sum_len = 0
        filenames = os.listdir(self.output_dir)
        for name in filenames:
            file_path = os.path.join(self.output_dir, name)
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    #print(line)
                    obj = json.loads(line)
                    if isinstance(obj["tgt"], str):
                        sum_len += len(obj["tgt"])
                        line_cnt += 1
                    else:
                        # beam search
                        for tgt in obj["tgt"]:
                            sum_len += len(tgt)
                        line_cnt += 1    
        # print avg len                
        print(f"total tokens: {sum_len}, ins count: {line_cnt}, avg tokens: {sum_len / line_cnt}")


class CVRDataset(FileIterableDataset):
    """ CVRDataset """
    def __init__(self,
                 input_file,
                 tokenizer,
                 batch_size,
                 drop_last,
                 max_input_length,
                 prompt_fmt):
        super().__init__(
                input_file,
                tokenizer,
                batch_size,
                drop_last,
                False,
                max_input_length)
        self.prompt_fmt = prompt_fmt
        self.cls, self.end = tokenizer.convert_tokens_to_ids(['[CLS]', '[END]'])

    def file_iter(self, file_list):
        """ file_iter """
        for file_path in file_list:
            with open(file_path, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    # fea = json.loads(example['fea_dict'])
                    # example['src'] = self.prompt_fmt.format(**fea)
                    yield example
    def decode_output(self, qids, outputs):
        """ decode output """
        output_list = []
        for _ in range(len(qids)):
            output_list.append({
                'qids': qids[_],
                'output': outputs[_]
            })

        return output_list

    def prepare_inputs(self, batched):
        """ prepare inputs """
        batched_qids = []
        batched_input_ids = []
        batched_input_lengths = []
        for example in batched:
            input_ids = self.tokenizer.encode(
                    example['src'],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_input_length - 2 # for [cls, end]
            )
            input_ids = [self.cls] + input_ids + [self.end]
            batched_qids.append(example['logkey'])
            batched_input_ids.append(input_ids)
            batched_input_lengths.append(len(input_ids))

        max_length = max(batched_input_lengths)
        # Right padding for trt-llm
        paddings = [
            np.ones(max_length - l, dtype=np.int32) * self.pad_id
            for l in batched_input_lengths
        ]
        batched_input_ids = [
            np.concatenate([x, pad]) for x, pad in zip(batched_input_ids, paddings)
        ]

        batch_size = len(batched_input_ids)
        batched_position_ids = np.zeros([batch_size, 2, max_length], dtype=np.int32)
        batched_position_ids[:, 0, :] = np.arange(max_length, dtype=np.int32)

        batched_input_ids = torch.tensor(np.array(batched_input_ids, dtype=np.int32))
        batched_input_lengths = torch.tensor(np.array(batched_input_lengths, dtype=np.int32))
        batched_position_ids = torch.tensor(batched_position_ids)

        return {
            'qids': batched_qids,
            'input_ids': batched_input_ids,
            'input_lengths': batched_input_lengths,
            'position_ids': batched_position_ids
        }
