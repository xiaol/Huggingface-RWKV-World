import torch
from ringrwkv.configuration_rwkv_world import RwkvConfig
from ringrwkv.rwkv_tokenizer import TRIE_TOKENIZER
from ringrwkv.modehf_world import RwkvForCausalLM


#model = RwkvForCausalLM.from_pretrained("RWKV-4-World-1.5B")
#model = RwkvForCausalLM.from_pretrained("RWKV-4-World-3B")
model = RwkvForCausalLM.from_pretrained("RWKV-4-World-0.4B")
tokenizer = TRIE_TOKENIZER('./ringrwkv/rwkv_vocab_v20230424.txt')
     
text = "你叫什么名字？"

question = f'Question: {text.strip()}\n\nAnswer:'

input_ids = tokenizer.encode(question)
#print(tokenizer.decode(input_ids))
input_ids = torch.tensor(input_ids).unsqueeze(0)
#print(input_ids)
#print(input_ids.shape) #[1,13]

# 将任意长度的token转为统一规格的tensor，进行emb，2048可看做ctx_max_length
#input_tensor = torch.zeros((1,2048)) #RWKB 1.5B
#print(input_tensor.shape)

#size = input_ids.shape[1]
#for i in range(size):
    #input_tensor[0][i] = input_ids[0][i]

#print(input_tensor)
#print(input_tensor.shape) #[1,2048]

#input_tensor = input_tensor.unsqueeze(1) #[1,1,2048]
#output = model.forward(inputs_embeds=input_tensor).last_hidden_state
#print(output)
#print(output.shape) #[1,1,2048]


# 得到回答
out = model.generate(input_ids,max_new_tokens=40)

#print(out[0])

outlist = out[0].tolist()

for i  in outlist:
    if i==0:
        outlist.remove(i)

#print(outlist)
answer = tokenizer.decode(outlist)

# answer = tokenizer.decode([10464, 11685, 19126, 12605, 11021, 10399, 12176, 10464, 16533, 10722,
#          10250, 10349, 17728, 18025, 10080, 16738, 17728, 10464, 17879, 16503])
# answer = tokenizer.decode([53648,    59,    33, 10464, 11017, 10373, 10303, 11043, 11860, 19156,
#           261, 40301,    59,    33, 12605, 13091, 10250, 10283, 10370, 12137,
#         13133, 15752, 16728, 16537, 13499, 11496, 19137, 13734, 13191, 11043,
#         11860, 10080])
print(answer)

#print(input_ids.shape)
#rwkvoutput = model.forward(input_ids=input_ids,labels=input_ids) #loss,logits,state,hidden_states,attentions
# print("loss:")
# print(rwkvoutput.loss)
# print("logits:")
# print(rwkvoutput.logits)
# print("state:")
# print(rwkvoutput.state)
#print("last_hidden_state:")
# print(rwkvoutput.last_hidden_state)
# print("attentions:")
# print(rwkvoutput.attentions)

