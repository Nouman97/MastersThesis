from __future__ import absolute_import, division, print_function
import copy, logging, math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

import os

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
logger = logging.getLogger(__name__)

def swish(x):
  return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
  def __init__(self, vis):
    super(Attention, self).__init__()
    self.vis = vis
    self.num_attention_heads = 12
    self.attention_head_size = int(768 / self.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = Linear(768, self.all_head_size)
    self.key = Linear(768, self.all_head_size)
    self.value = Linear(768, self.all_head_size)

    self.out = Linear(768, 768)
    self.attn_dropout = Dropout(0.0)
    self.proj_dropout = Dropout(0.0)

    self.softmax = Softmax(dim = -1)

  def transpose_for_scores(self, x):
    # perhaps -> batch_size x  patches x embeddings => batch_size x heads x patches x new_embedding_size
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_probs = self.softmax(attention_scores)
    weights = attention_probs if self.vis else None
    attention_probs = self.attn_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)

    # perhaps -> batch_size x heads x patches x new_embedding_size => batch_size x  patches x heads x new_embedding_size
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    attention_output = self.out(context_layer)
    attention_output = self.proj_dropout(attention_output)

    return attention_output, weights

class Mlp(nn.Module):
  def __init__(self):
    super(Mlp, self).__init__()
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)
    self.act_fc = ACT2FN["gelu"]
    self.dropout = Dropout(0.1)

    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.normal_(self.fc1.bias, std = 1e-6)
    nn.init.normal_(self.fc2.bias, std = 1e-6)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act_fc(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x

class Embeddings(nn.Module):
  def __init__(self, img_size, in_channels = 3):
    super(Embeddings, self).__init__()
    self.hybrid = None
    img_size = _pair(img_size)

    # 1 x 1    
    patch_size = (16, 16)
    n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
    self.hybrid = False

#    self.hybrid_model = ResNetV2(block_units = (3, 4, 9), width_factor = 1)
#    in_channels = self.hybrid_model.width * 16

    #print(patch_size)
    self.patch_embeddings = Conv2d(in_channels = in_channels, out_channels = 768, kernel_size = patch_size, stride = patch_size)
    self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

    self.dropout = Dropout(0.1)

  def forward(self, x):
    features = None
    #print(x.shape)
    #x, features = self.hybrid_model(x)
    # _ = [print("Features: ", i.shape) for i in features]
    #print(x.shape)
    x = self.patch_embeddings(x)
    #print(x.shape)
    x = x.flatten(2)
    #print(x.shape)
    x = x.transpose(-1, -2)
    #print(x.shape)
    embeddings = x + self.position_embeddings
    embeddings = self.dropout(embeddings)

    return embeddings, features


class Block(nn.Module):
  def __init__(self, vis):
    super(Block, self).__init__()
    self.hidden_size = 768
    self.attention_norm = LayerNorm(768, eps = 1e-6)
    self.ffn_norm = LayerNorm(768, eps = 1e-6)
    self.ffn = Mlp()
    self.attn = Attention(vis = True)

  def forward(self, x):
    h = x
    x = self.attention_norm(x)
    #print("Attention Input: ", x.shape)
    x, weights = self.attn(x)
    #print("Attention Weights: ", weights.shape)
    #print("Attention Result: ", x.shape)
    x = x + h
    h = x
    x = self.ffn_norm(x)
    x = self.ffn(x)
    #print("FFN Result: ",x.shape)
    x = x + h
    return x, weights

  def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
        
class Encoder(nn.Module):
  def __init__(self, vis, num_layers = 12):
    super(Encoder, self).__init__()
    self.vis = vis
    self.layer = nn.ModuleList()
    self.encoder_norm = LayerNorm(768, eps = 1e-6)
    for _ in range(num_layers):
      layer = Block(vis)
      self.layer.append(copy.deepcopy(layer))

  def forward(self, hidden_states):
    attn_weights = []
    count = 1
    #for layer_block in self.layer:
      #print("Block: {}".format(count))
      #count += 1
      #hidden_states, weights = layer_block(hidden_states)
      #if self.vis:
        #attn_weights.append(weights)

    features = []
    for layer_block in self.layer:
      x, weights = layer_block(hidden_states)
      features.append(x)
      if self.vis:
        attn_weights.append(weights)

    features = features[:-1]

    encoded = self.encoder_norm(x)

    features.append(encoded)
    return features, attn_weights

class Transformer(nn.Module):
  def __init__(self, img_size, vis, num_layers = 12):
    super(Transformer, self).__init__()
    self.embeddings = Embeddings(img_size = img_size)
    self.encoder = Encoder(vis, num_layers)

  def forward(self, input_ids):
    embedding_output, features = self.embeddings(input_ids)
    encoded, attn_weights = self.encoder(embedding_output)
    return encoded, attn_weights, features

def load_from(model, weights):
  with torch.no_grad():
    res_weight = weights
    model.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
    model.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

    model.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
    model.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

    posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

    posemb_new = model.embeddings.position_embeddings
    if posemb.size() == posemb_new.size():
        model.embeddings.position_embeddings.copy_(posemb)
    elif posemb.size()[1]-1 == posemb_new.size()[1]:
        posemb = posemb[:, 1:]
        model.embeddings.position_embeddings.copy_(posemb)
    else:
        logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
        ntok_new = posemb_new.size(1)
        if model.classifier == "seg":
            _, posemb_grid = posemb[:, :1], posemb[0, 1:]
        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb = posemb_grid
        model.embeddings.position_embeddings.copy_(np2th(posemb))

    # Encoder whole
    for bname, block in model.encoder.named_children():
        for uname, unit in block.named_children():
            unit.load_from(weights, n_block=uname)
                  
def ViT(load_pretrained = True, num_layers = 12, vis = True):
    if load_pretrained == True:
        try:
            try:
                os.mkdir("imagenet21k")
            except Exception as e:
                print(e)
            os.system("wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz >/dev/null 2>&1")
            os.system("wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz >/dev/null 2>&1")
            os.system("mv R50+ViT-B_16.npz imagenet21k/R50+ViT-B_16.npz >/dev/null 2>&1")
            os.system("mv ViT-B_16.npz imagenet21k/ViT-B_16.npz >/dev/null 2>&1")
        except Exception as e:
            print(e)
    net = Transformer(224, vis = vis)
    if load_pretrained == True:
        load_from(net, weights=np.load("./imagenet21k/ViT-B_16.npz"))
    return net                  
