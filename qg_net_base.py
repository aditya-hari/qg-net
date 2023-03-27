# -*- coding: utf-8 -*-
"""qg_net_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10sUaVcsk-kvNx5o5WTd7cGOVKP3vYIry
"""

import torchtext 
from torchtext.legacy import data
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd 
import numpy as np 
import tqdm
from copy import deepcopy

#root = 'drive/MyDrive/adv_nlp_project'
root = '.'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# from google.colab import drive
# drive.mount('/content/drive')

"""## Getting Features """

def get_src_and_feat(line):
  token_level_split = [tok.split('￨') for tok in line.split()]
  sent = ' '.join([i[0] for i in token_level_split])
  features = [[],[],[],[]]
  for token in token_level_split:
    for i, feature in enumerate(token[1:]):
      features[i].append(feature)
  return sent, features

for splt in ['train','dev','test']:
  source = open(f'{root}/data/{splt}/squad.corenlp.filtered.contents.features.1sent.txt', 'r').readlines()
  target = open(f'{root}/data/{splt}/squad.corenlp.filtered.questions.txt', 'r').readlines()

  result_dict = {
      'src':[],
      'feat_0':[],
      'feat_1':[],
      'feat_2':[],
      'feat_3':[],
      'trg':[]
  }

  for src, trg in zip(source, target):
    sent, features = get_src_and_feat(src)
    result_dict['src'].append(sent)
    result_dict['trg'].append(trg)
    for i, feat in enumerate(features):
      result_dict[f'feat_{i}'].append(" ".join(feat))

  result_df = pd.DataFrame.from_dict(result_dict)
  result_df.to_csv(f'{splt}.tsv', sep ='\t', index=False)

"""## Dataset Preparation"""

def tokenizer(text):
  return text.split(" ")

SRC = data.Field(sequential=True, tokenize=tokenizer, include_lengths = True, init_token = '<sos>', eos_token = '<eos>', lower=True)
TRG = data.Field(sequential=True, tokenize=tokenizer, include_lengths = True, init_token = '<sos>', eos_token = '<eos>', lower=True)
FEAT_0 = data.Field(sequential=True, tokenize=tokenizer, init_token = '-', eos_token = '-')
FEAT_1 = data.Field(sequential=True, tokenize=tokenizer, init_token = '-', eos_token = '-')
FEAT_2 = data.Field(sequential=True, tokenize=tokenizer, init_token = '-', eos_token = '-')
FEAT_3 = data.Field(sequential=True, tokenize=tokenizer, init_token = '-', eos_token = '-')

train, val, test = data.TabularDataset.splits(
        path='./', train='train.tsv', validation = 'dev.tsv', test = 'test.tsv', format='tsv',
        fields=[('src', SRC), 
                ('feat_0', FEAT_0), ('feat_1', FEAT_1), ('feat_2', FEAT_2), ('feat_3', FEAT_3), 
                ('trg', TRG)], 
        skip_header = True)

FEAT_0.build_vocab(train)
FEAT_1.build_vocab(train)
FEAT_2.build_vocab(train)
FEAT_3.build_vocab(train)

FEATS = [FEAT_0, FEAT_1, FEAT_2, FEAT_3]

SRC.build_vocab(train.src, train.trg, vectors="glove.6B.300d")
TRG.vocab = SRC.vocab

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), sort_key=lambda x: len(x.src), sort_within_batch = True,
        batch_sizes=(16, 16, 1), device=device)

"""### Model """

class RNNEncoder(nn.Module):
  def __init__(self, embedding, feature_vocab_sizes, padding_idxs, num_hidden, num_layers, bidirectional, dropout = 0.3):
    super(RNNEncoder, self).__init__()
    self.embeddings = nn.ModuleList([embedding])
    self.embeddings.extend(self.make_embedding(feature_vocab_sizes, padding_idxs))
    self.input_size = sum([emb.embedding_dim for emb in self.embeddings])
    self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = num_hidden, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional)
    self.hidden_reduce = nn.Linear(num_hidden*2, num_hidden)
    self.cell_reduce = nn.Linear(num_hidden*2, num_hidden)

  def forward(self, sent, feats, src_len):
    inputs =  [sent]+feats
    embedded = [emb(feat) for (emb, feat) in zip(self.embeddings, inputs)]
    embedded = torch.cat(embedded, dim = 2)
    packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))
    output, (hidden, cell) = self.rnn(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(output) 
    
    hidden = self.hidden_reduce(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
    cell = self.cell_reduce(torch.cat((cell[0:1], cell[1:2]), dim=2))

    return output, (hidden, cell) 
  
  def make_embedding(self, feature_vocab_sizes, padding_idxs):
    input_dims = feature_vocab_sizes
    output_dims = [int(sz**0.7) for sz in feature_vocab_sizes]
    embeddings = nn.ModuleList([nn.Embedding(i_dim, o_dim, padding_idx = p_idx) for (i_dim, o_dim, p_idx) in zip(input_dims, output_dims, padding_idxs)])
    return embeddings

class RNNDecoder(nn.Module):
  def __init__(self, embedding, num_hidden, num_layers, dropout):
    super(RNNDecoder, self).__init__()
    self.embedding = embedding
    self.vocab_size = self.embedding.num_embeddings
    self.embedding_size = self.embedding.embedding_dim
    self.rnn = nn.LSTM(input_size = self.embedding_size, hidden_size = num_hidden, num_layers = num_layers)
    self.dropout = nn.Dropout(dropout)

    # Attention 
    self.W_h = nn.Parameter(torch.randn((num_hidden*2, num_hidden)))
    self.W_s = nn.Parameter(torch.randn((num_hidden, num_hidden)))
    self.v = nn.Parameter(torch.randn((num_hidden, 1)))
    self.b_attn = nn.Parameter(torch.randn((1, num_hidden)))

    # Pointer
    self.V = nn.Linear(num_hidden*3, self.vocab_size)
    # self.V_dash = nn.Linear(num_hidden, self.vocab_size)
    self.W_z = nn.Linear(num_hidden, 1)

  def forward(self, inputs, encoder_outputs, encoder_word_idx, hidden = None):
    inputs = inputs.unsqueeze(0)
    embedded = self.embedding(inputs)
    decoder_outputs, hidden = self.rnn(embedded, hidden)
    decoder_outputs = self.dropout(decoder_outputs)
    # Attention
    x1 = encoder_outputs.matmul(self.W_h)
    x2 = decoder_outputs.expand(encoder_outputs.size(0), -1, -1).matmul(self.W_s)
    bias = self.b_attn.unsqueeze(0).repeat(encoder_outputs.size(0), 1, 1)
    res = torch.tanh(x1 + x2 + bias).matmul(self.v).squeeze(2)
    attn = F.softmax(res, 0).transpose(0, 1)
    # Pointer 
    encoder_outputs = encoder_outputs.transpose(0, 1)
    context = attn.unsqueeze(1).bmm(encoder_outputs).squeeze(1)
    global_state = torch.cat((decoder_outputs.squeeze(0), context), 1)
    #e = torch.sigmoid(self.V(global_state))
    #p_vocab = torch.softmax(self.V_dash(e), 1)
    p_vocab =  torch.selu(self.V(global_state))
    
    p_source = attn 
    p = torch.sigmoid(self.W_z(decoder_outputs))
    p = p.squeeze(0)
    que_vocab_distr = p * p_vocab
    inp_vocab_distr = (1 - p) * p_source

    extended_vocab_distr = que_vocab_distr.scatter_add_(1, encoder_word_idx.transpose(0, 1), inp_vocab_distr)
    
    return extended_vocab_distr, hidden, p.squeeze(), attn.transpose(0, 1)

class QGNet(nn.Module):
  def __init__(self, vocab_size, feature_vocabs, emd_dim, pretrained_weights, num_hidden, num_layers, pad_idxs, unk_idx, dropout = 0.3):
    super(QGNet, self).__init__()
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, emb_dim).from_pretrained(pretrained_weights, padding_idx = padding_idxs[0])
    self.encoder = RNNEncoder(self.embedding, feature_vocabs, padding_idxs[1:], 600, 2, True, dropout)
    self.decoder = RNNDecoder(self.embedding, 600, 2, 0.5)
    self.unk_idx = unk_idx
  
  def forward(self, src, src_lens, feats, max_seq_len, trg = None):
    outputs = []
    encoder_outputs, _ = self.encoder(src, feats, src_lens)
    hidden = None 
    if(trg != None):
      trg_len = trg.shape[0]
      inputs = trg[0, :]
      for t in range(1, trg_len):
        dist, hidden, prob, attn = self.decoder(inputs, encoder_outputs, src, hidden)
        outputs.append(dist)
        inputs = trg[t, :]
    else:
      batch_size = src.shape[1]
      inputs = (torch.ones(batch_size, device=device) * SRC.vocab.stoi['<sos>']).long()
      for i in range(1, max_seq_len):
        dist, hidden, prob, attn = self.decoder(inputs, encoder_outputs, src, hidden)
        outputs.append(dist)
        inputs = torch.argmax(dist, 1).long()

    return torch.stack(outputs)

def train_step(model, criterion, optimizer, batch):
    optimizer.zero_grad()
    src, src_lens = batch.src 
    feats = [batch.feat_0, batch.feat_1, batch.feat_2, batch.feat_3]
    trg, trg_lens = batch.trg 

    if(src.shape[0]!=feats[0].shape[0]):
      return 0, 0 

    dist = model(src, src_lens, feats, None, trg)

    dist = dist.view(-1, dist.size(2))
    trg = trg[1:].view(-1)
    loss = criterion(dist, trg)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    return loss.item(), grad_norm

def eval_step(model, criterion, batch):
    src, src_lens = batch.src 
    feats = [batch.feat_0, batch.feat_1, batch.feat_2, batch.feat_3]
    trg, trg_lens = batch.trg 
    max_trg_len = trg_lens.max().item()

    if(src.shape[0]!=feats[0].shape[0]):
      return 0
      
    dist = model(src, src_lens, feats, max_trg_len)
    dist = dist.view(-1, dist.size(2))
    trg = trg[1:].view(-1)

    loss = criterion(dist, trg)
    return loss.item()

def train_model(model, train_iter, val_iter, epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
  criterion = torch.nn.CrossEntropyLoss(ignore_index = padding_idx)

  best_model_weights = None 
  best_valid_loss = np.inf

  for epoch in range(1, epochs+1):
    epoch_train_loss = 0
    epoch_valid_loss = 0

    model.train()
    with(tqdm.tqdm(train_iter, unit="batch")) as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        loss, _ = train_step(model, criterion, optimizer, batch)
        epoch_train_loss += loss
        tepoch.set_postfix(loss=loss)  
    epoch_train_loss = epoch_train_loss/len(train_iter)

    model.eval()
    with(tqdm.tqdm(val_iter, unit="batch")) as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        loss = eval_step(model, criterion, batch)
        epoch_valid_loss += loss
        tepoch.set_postfix(loss=loss)  
    epoch_valid_loss = epoch_valid_loss/len(val_iter)

    if(epoch_valid_loss < best_valid_loss):
      best_valid_loss = epoch_valid_loss 
      best_model_weights = deepcopy(model.state_dict())

    print(f'\n Epoch {epoch} ; Train loss: {epoch_train_loss}; Valid. loss: {epoch_valid_loss} \n')

    if(epoch%3 == 0):
      save_dict = {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
      }
      torch.save(save_dict, f'./checkpoints/model_{epoch}.pt')

  return best_model_weights

vocab_size = len(SRC.vocab.stoi)
emb_dim = 300
feature_vocab_sizes = [len(feat.vocab.stoi) for feat in FEATS]
padding_idxs = [field.vocab.stoi['<pad>'] for field in FEATS]
padding_idx = SRC.vocab.stoi['<pad>']
unk_idx = SRC.vocab.stoi['<unk>']
pretrained_weights = SRC.vocab.vectors

model = QGNet(vocab_size, feature_vocab_sizes, emb_dim, pretrained_weights, 600, 2, padding_idxs, unk_idx)
model.to(device)

best_weights = train_model(model, train_iter, val_iter, 10)
torch.save(best_weights, './best_model.pt')