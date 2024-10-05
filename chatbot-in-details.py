import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math
import torch; torch.set_printoptions(precision=3)
import numpy as np
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from download import download_LaMini_model; download_LaMini_model()

temperature = 1.0
def set_temperature(value):
    global temperature
    temperature = value

def layer_normalization(self, hidden_states):
    hidden_states = hidden_states.clone()
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    variance_epsilon = 1e-6
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    normed_hidden_states = self.weight * hidden_states
    return normed_hidden_states
    
def attention(self, hidden_states, mask=None, key_value_states=None):
    batch_size, seq_length = hidden_states.shape[:2]
    key_length = seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states): # projection
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states): # reshape
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    query_states = shape(self.q(hidden_states)) 
    key_states = shape(self.k(hidden_states if key_value_states is None else key_value_states))
    value_states = shape(self.v(hidden_states if key_value_states is None else key_value_states))
    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    
    if not self.has_relative_attention_bias:
        position_bias = torch.zeros((1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype)
    else:
        position_bias = self.compute_bias(seq_length, key_length, device=scores.device)

    if mask is not None:
        position_bias = position_bias + mask 

    scores += position_bias
    attn_weights = torch.nn.functional.softmax(scores.float()/temperature, dim=-1).type_as(scores)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=False) 
    
    attn_output = unshape(torch.matmul(attn_weights, value_states))  
    attn_output = self.o(attn_output)

    return attn_output, position_bias

def self_attention(self, hidden_states, mask=None):
    return attention(self, hidden_states, mask)

def cross_attention(self, hidden_states, mask=None, key_value_states=None):
    return attention(self, hidden_states, mask, key_value_states)

def GELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
def FFN(self, hidden_states):
    forwarded_states = layer_normalization(self.layer_norm, hidden_states)
    hidden_gelu = GELU(self.DenseReluDense.wi_0(forwarded_states))
    hidden_linear = self.DenseReluDense.wi_1(forwarded_states)
    forwarded_states = hidden_gelu * hidden_linear
    forwarded_states = self.DenseReluDense.dropout(forwarded_states)
    forwarded_states = self.DenseReluDense.wo(forwarded_states)
    hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states

def T5block(self, hidden_states, mask, encoder_hidden_states=None, encoder_attention_mask=None):
    normed_hidden_states = layer_normalization(self.layer[0].layer_norm, hidden_states)
    attention_output = self_attention( self.layer[0].SelfAttention, 
        normed_hidden_states, 
        mask
    )
    hidden_states = hidden_states + self.layer[0].dropout(attention_output[0])
    attention_outputs = (attention_output[1],)  
    if self.is_decoder and encoder_hidden_states is not None:
        normed_hidden_states = layer_normalization(self.layer[1].layer_norm, hidden_states)
        cross_attention_output = cross_attention( self.layer[1].EncDecAttention, 
            normed_hidden_states, 
            encoder_attention_mask, 
            encoder_hidden_states
        )
        hidden_states = hidden_states + self.layer[1].dropout(cross_attention_output[0])
        attention_outputs += (cross_attention_output[1],)
    hidden_states = FFN(self.layer[-1], hidden_states)
    return (hidden_states,) + attention_outputs

checkpoint = "./LaMini/"  # LaMini-Flan-T5-248M
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device='cpu')
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cpu')
set_temperature(1.0)

input_text = "What is the capital of the USA?"
print("Question:",input_text)

# tokenizer - encode
tokens = tokenizer.encode(input_text, return_tensors="pt") # shape [1, 9]

# add start token
pad = base_model.config.pad_token_id # 0 
eos = base_model.config.eos_token_id # 1
start = torch.tensor([[pad]])
input_tokens = torch.concatenate([start,tokens],dim=1) # shape [1, 10]

# embed
embed = base_model.shared.weight[input_tokens]  # shape [1, 10, 768]

# encode
def encode(x, mask=None):
    for block in base_model.encoder.block:
        x, mask = T5block(block, x, mask)
    return layer_normalization(base_model.encoder.final_layer_norm, x)

hidden = encode(embed) # shapes [1, 10, 768], [1, 12, 10, 10]

# generate
output_tokens = start
while True:
    
    # embed
    embed = base_model.shared.weight[output_tokens] # shape [1, 1, 768]
    
    # decode
    def decode(x, mask, crossx, crossmask):
        for block in base_model.decoder.block:
            x, mask, crossmask = T5block(block, x, mask, crossx, crossmask)
        return layer_normalization(base_model.decoder.final_layer_norm, x)

    output = decode(embed, None, hidden, None) # shape [1, N, 768]
    
    # wipe out
    logits = torch.matmul(output[0], base_model.lm_head.weight.t())
    next_token = torch.argmax(logits[-1,:]) # 0-32127 
    
    # add the next token
    output_tokens = torch.concatenate([output_tokens,torch.tensor([[next_token]])],dim=1)
    
    #print('  ', tokenizer.decode(output_tokens[0], skip_special_tokens=False))

    if next_token == eos:
        break

# tokenizer - decode
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("Answer:", output_text) # 'Washington, D.C.'