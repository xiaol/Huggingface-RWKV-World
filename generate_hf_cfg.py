import torch
#from peft import PeftModel
import transformers
import gradio as gr

from ringrwkv.configuration_rwkv_world import RwkvConfig
from ringrwkv.rwkv_tokenizer import TRIE_TOKENIZER
from ringrwkv.modehf_world import RwkvForCausalLM
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper)
import torch.nn.functional as F

class CFGLogits(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
    """

    def __init__(self, guidance_scale, uncond, model):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.out = None
        self.rescale_factor = 1 #rescale_factor

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        if self.out is None:
            self.out = self.model(self.uncond, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                # past_key_values=self.out.past_key_values,
                state = self.out.state
            )
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out

        # out = F.log_softmax(out, dim=-1)
        # return 0.7 * out + 0.3 * scores





if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#放在本地工程根目录文件夹


#model = RwkvForCausalLM.from_pretrained("RWKV-4-World-7B", torch_dtype=torch.bfloat16)
model = RwkvForCausalLM.from_pretrained("RWKV-4-World-7B", torch_dtype=torch.float16)
tokenizer = TRIE_TOKENIZER('./ringrwkv/rwkv_vocab_v20230424.txt')


#model= PeftModel.from_pretrained(model, "./lora-out")
model = model.half()
# model = torch.compile(model) #need pytorch 2.0 and linux
model = model.to(device)


def evaluate(
    instruction,
    temperature=1.3,
    top_p=0.4,
    # top_k = 0.1,
    cfg =3,
    penalty_alpha = 0.4,
    max_new_tokens=128,
):
    
    prompt = f'{instruction.strip()}'
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    #out = model.generate(input_ids=input_ids.to(device),max_new_tokens=40)
    out = model.generate(input_ids=input_ids,
                        #  temperature=temperature,
                        #  top_p=top_p,top_k=top_k,
                         penalty_alpha=penalty_alpha,
                         max_new_tokens=max_new_tokens,
                         logits_processor=LogitsProcessorList([
                            # inputs_cfg usually is the last token of the prompt but there are
                                # possibilities of negative prompting that are explored in the paper
                                            #CFGLogits(1.5, neg_prompt.to(device), model),
                                            CFGLogits(cfg, input_ids, model),
                                            TemperatureLogitsWarper(temperature),
                                            TopPLogitsWarper(top_p),
                                            ]),do_sample=True,)
    outlist = out[0].tolist()
    for i  in outlist:
        if i==0:
            outlist.remove(i)
    answer = tokenizer.decode(outlist)
    return answer.strip()
    #return answer.split("### Response:")[1].strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=2, value=1.4, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.3, label="Top p"),
        gr.components.Slider(minimum=-5, maximum=5, step=0.1, value=3, label="cfg"),
        gr.components.Slider(minimum=0, maximum=1, step=0.1, value=0.4, label="penalty_alpha"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="RWKV-World-alpaca",
    description="RWKV,Easy In HF.",
).launch()