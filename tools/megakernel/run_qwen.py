import argparse
import os
import tensorrt as trt
import torch
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm._utils import torch_dtype_to_trt
from transformers import AutoTokenizer
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, default='./mk_engine')
    parser.add_argument('--tokenizer_dir', type=str, default='./Qwen3-1.7B')
    parser.add_argument('--engine_name', type=str, default='rank0.engine')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate')
    parser.add_argument('--prompt', type=str, default='讲一个长故事', help='Input prompt for generation')
    parser.add_argument('--batch_size', type=int, default=2, help='Use max batch size generation')
    return parser.parse_args()

class ModelRunner(object):
    """ model runner """
    def __init__(self, args):
        self.max_batch_size = args.batch_size
        self.max_seq_len = 128
        self.num_hidden_layers = 28
        self.num_kv_heads = 8
        self.head_dim = 128
        self.init_buffer()
        self.create_session(args)
        self.input_lengths = None

    def create_session(self, args):
        engine_path = os.path.join(args.engine_dir, args.engine_name)
        # 加载引擎
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        # 创建Session
        self.session = Session.from_serialized_engine(engine_buffer)

    def make_bs_params(self, seq_lens):
        bs = len(seq_lens)
        offset = 0
        for i in range(bs):
            self.bs_params[i, 0] = offset
            self.bs_params[i, 1] = seq_lens[i]
            self.bs_params[i, 2] = 0
            offset += seq_lens[i]
        self.seq_lens = seq_lens.clone()

    def update_bs_params(self, new_tokens_num):
        bs = len(self.seq_lens)
        offset = 0
        for i in range(bs):
            self.bs_params[i, 0] = offset
            self.bs_params[i, 1] = 1
            self.bs_params[i, 2] = self.seq_lens[i]
            offset += 1
            self.seq_lens[i] += new_tokens_num

    def init_buffer(self):
        def make_buffer(shape, buffer_dtype=torch.bfloat16):
            return torch.zeros(shape, device="cuda", dtype=buffer_dtype)
        nlayers_mbatchsize = self.num_hidden_layers * self.max_batch_size
        self.k_cache = make_buffer([nlayers_mbatchsize, self.max_seq_len, self.num_kv_heads, self.head_dim])
        self.v_cache = self.k_cache.clone()
        self.bs_params = torch.zeros([self.max_batch_size, 3], dtype=torch.int32, device="cuda")

    def infer_shape(self, inputs):
        """ infer shape """
        input_infos = []
        for k, v in inputs.items():
            #print(f"{k} {v.shape} {v.dtype}")
            input_infos.append(TensorInfo(k, torch_dtype_to_trt(v.dtype), v.shape))
        output_info = self.session.infer_shapes(input_infos)

        # 创建输出张量
        outputs = {}
        for t in output_info:
            out_shape = []
            for dim in t.shape:
                if dim == -1:
                    out_shape.append(1)  # 默认值
                else:
                    out_shape.append(dim)
            outputs[t.name] = torch.empty(
                tuple(out_shape),
                dtype=torch.float32 if t.name == 'logits' else torch.bfloat16,
                device="cuda",
            )
        return outputs

    def get_input_and_output(self, input_ids, input_lengths):
        batch_size = input_lengths.shape[0]
        token_num = torch.sum(input_lengths)
        if token_num == batch_size:
            self.update_bs_params(1)
        else: # encoder
            self.make_bs_params(input_lengths)
        bs_params = self.bs_params[:batch_size, :]
        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "k_cache": self.k_cache,
            "v_cache": self.v_cache,
            "bs_params": bs_params,
        }
        outputs = self.infer_shape(inputs)
        # 运行推理
        return inputs, outputs
    
    def run(self, input_ids, input_lengths):
        """ run encoder or decoder """
        input_ids = input_ids.view(-1)
        #print("input_ids=", input_ids, ", input_lengths=", input_lengths)
        inputs, outputs = self.get_input_and_output(input_ids, input_lengths)
        stream = torch.cuda.current_stream().cuda_stream
        #print("start infer")
        torch.cuda.synchronize()
        self.session.run(inputs, outputs, stream)
        torch.cuda.synchronize()
        # print output
        # for k, v in outputs.items():
        #     print(f"{k}=", v.shape, v)
        return outputs


def main():
    args = parse_arguments()
    print("args:", args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    input_ids_cpu = tokenizer(args.prompt, add_special_tokens=True)["input_ids"] 

    input_ids = torch.tensor(input_ids_cpu * args.batch_size, dtype=torch.int32).to('cuda')
    seq_lens = [len(input_ids_cpu)] * args.batch_size
    input_lengths = torch.tensor(seq_lens, dtype=torch.int32)
    print("input_ids=", input_ids, ", input_lengths=", input_lengths)

    batch_size = input_lengths.shape[0]
    runner = ModelRunner(args)
    outputs = runner.run(input_ids, input_lengths)
    logits = outputs['logits']
    output_ids = torch.argmax(logits, dim=-1)
    #print("output_ids=", output_ids)
    output_tokens = torch.zeros(
        batch_size, 
        args.max_new_tokens, 
        device=input_ids.device, 
        dtype=torch.int32
    )
    offset = 0
    for bs in range(batch_size):
        offset += input_lengths[bs].item()    
        output_tokens[bs][0] = output_ids[offset - 1]
    #print("output_tokens=", output_tokens)

    input_token_pos = 0
    input_ids = input_ids[: batch_size]
    for step in range(1, args.max_new_tokens):
        for bs in range(batch_size):
            input_ids[bs] = output_tokens[bs][input_token_pos]
        input_lengths.fill_(1)
        outputs = runner.run(input_ids, input_lengths)
        logits = outputs['logits']
        output_ids = torch.argmax(logits, dim=-1)
        #print("output_ids=", output_ids)
        output_tokens[:, input_token_pos + 1 : input_token_pos + 2] = output_ids.view(batch_size, -1)
        input_token_pos = input_token_pos + 1
    ## output tokens
    to_cpu = output_tokens.cpu()
    print("Output ids: ", to_cpu)
    print("Output text: ", tokenizer.batch_decode(to_cpu))
    
if __name__ == '__main__':
    main()
