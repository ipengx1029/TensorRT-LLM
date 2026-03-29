""" in-flight
"""


import time
import json
import argparse
import threading
import multiprocessing
import asyncio
from tensorrt_llm.hlapi import LLM, ModelConfig
from tensorrt_llm.hlapi.utils import SamplingConfig
from tensorrt_llm.executor import GenerationRequest
from tqdm import tqdm


class InFlighter():
    """ InFlighter """
    def __init__(self,
                 engine_dir,
                 tokenizer,
                 dataloader,
                 datadumper,
                 **_additional_options):
        """ init """
        self.llm = LLM(ModelConfig(model_dir=engine_dir),
                       tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.update_interval = _additional_options.pop('update_interval', 8)
        self.beam_width = _additional_options.pop('num_beams', 1)
        self.max_concurrent_tasks = _additional_options.pop('max_concurrent_tasks', 1024)
        self.example_num = 0

        self.sampling_config = self.llm.get_default_sampling_config()
        setattr(self.sampling_config, 'beam_width', self.beam_width)
        setattr(self.sampling_config, 'max_new_tokens', _additional_options.pop('max_output_len', None))
        for k in self.sampling_config.get_attr_names():
            if k in _additional_options:
                setattr(self.sampling_config, k, [_additional_options[k]])
        self.exception = ''
        self.datadumper = datadumper
        self.tqdm = tqdm(ncols=50)

    def run(self):
        """ run """
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.generate())
        if self.exception:
            self.llm.shutdown()
            raise self.exception
    def close(self):
        """ close """
        self.llm.shutdown()
        self.tqdm.close()

    async def generate(self):
        """ producer """
        async def task_func(qid, input_ids):
            """ task function """
            future = self.llm._executor.submit(
                    GenerationRequest(
                        input_ids,
                        streaming=self.beam_width == 1,
                        tokenizer=self.tokenizer,
                        sampling_config=self.sampling_config,
                        exclude_input_from_output=True))
            result = await future.aresult()
            self.handle_result(qid, result)

        tasks = []
        for batch_dict in self.dataloader:
            for qid, input_ids in zip(batch_dict['qids'], batch_dict['input_ids']):
                tasks.append(asyncio.create_task(task_func(qid, input_ids)))
                if len(tasks) >= self.max_concurrent_tasks:
                    done, pending = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED)
                    try:
                        [x.result() for x in done]
                    except Exception as e:
                        self.exception = e
                        return
                    tasks = [t for t in pending]

        if tasks:
            done = await asyncio.gather(*tasks, return_exceptions=True)
            for x in done:
                if isinstance(x, Exception):
                    self.exception = x
                    return

    def handle_result(self, qid, result):
        """ handle_result """
        src = result.generation_request.input_ids.tolist()
        tgt = []
        if self.beam_width == 1:
            l = [result.token_ids]
        else:
            l = result.token_ids
        for token_ids in l:
            tgt.append(token_ids)

        self.example_num += 1   
        self.datadumper(qid, src, tgt)
        if self.example_num % self.update_interval == 0:
            self.tqdm.update(self.update_interval)


def main(engine_dir: str,
         input_file: str,
         batch_size: int,
         num_beams: int,
         max_input_length: int):
    """ main function """
    from zeus.tokenization_zeus import ZeusTokenizer
    from reader import FileIterableDataset
    from torch.utils.data import DataLoader

    tokenizer = ZeusTokenizer.from_pretrained('zeus/vocab.txt')
    dataset = FileIterableDataset(input_file=input_file,
                                  tokenizer=tokenizer,
                                  batch_size=batch_size,
                                  max_input_length=max_input_length,
                                  drop_last=False,
                                  add_special_tokens=True)
    dataloader = DataLoader(dataset, batch_size=None, pin_memory=True)

    queue = multiprocessing.Queue()
    def func(q):
        while q.get() != 'STOP':
            pass
    p = multiprocessing.Process(target=func, args=(queue,))
    p.daemon = True
    p.start()

    def woker(out):
        queue.put(out)
    flighter = InFlighter(engine_dir,
                          tokenizer,
                          dataloader,
                          woker,
                          num_beams=num_beams)
    flighter.run()
    queue.put('STOP')
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    parser.add_argument("--engine_dir", required=True, help="Engine directory.")
    parser.add_argument("--input_file", required=True, help="Input file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length.")

    # 解析命令行参数
    args = parser.parse_args()
    main(**vars(args))