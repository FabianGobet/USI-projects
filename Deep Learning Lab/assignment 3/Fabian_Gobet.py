'''
Generation of news titles using LSTM
Data taken from: https://www.kaggle.com/datasets/rmisra/news-category-dataset/
Student: Fabian Gobet
'''
## Packages
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordTokenizer

PATH = "./"

## Classes
class Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_sequences: list, word_to_int_dict: dict):
      self.data = []
      for sequence in tokenized_sequences:
        tokenized = [word_to_int_dict.get(word, word_to_int_dict['PAD'])  for word in sequence]
        self.data.append(tokenized)

  def __len__(self):
      return len(self.data)

  def __getitem__(self, ix):
      d = self.data[ix]
      return torch.tensor(d[:-1]), torch.tensor(d[1:])


class myNet(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers,dropout=0):
      super(myNet, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.embed = nn.Embedding(vocab_size, embed_size)
      self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
      self.linear = nn.Linear(hidden_size, vocab_size)
      self.drop = nn.Dropout(dropout)

  def forward(self, x, h):
      x = self.embed(x)
      out, (h, c) = self.lstm(x, h)
      out = self.drop(out)
      out = self.linear(out)
      return out, (h, c)

  def init_state(self, b_size=1):
      h = c = torch.zeros(self.num_layers, b_size, self.hidden_size)
      return h, c


## Functions
def save_to_pickle(object_var, path):
  with open(path, 'wb') as handle:
    pickle.dump(object_var, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_from_pickle(path):
  with open(path, 'rb') as handle:
    return pickle.load(handle)



def collate_fn(batch,pad_idx):
  data, targets = zip(*batch)

  padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_idx)
  padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

  return padded_data, padded_targets



def random_sample_next(model, x, prev_state, topk=5, uniform=True):
  out, state = model(x, prev_state)
  last_out = out[0, -1, :]

  topk = topk if topk else last_out.shape[0]
  top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)

  p = None if uniform else F.softmax(top_logit.cpu().detach(), dim=-1).numpy()
  sampled_ix = np.random.choice(top_ix.cpu(), p=p)

  return sampled_ix, state



def sample_argmax(model, x, prev_state,topk=None, uniform=None):
  out, state = model(x, prev_state)
  last_out = out[0, -1, :]
  top_logit, top_ix = torch.topk(last_out, k=1, dim=-1)
  return top_ix, state



def sample(model, seed_words, sample_function, device, topk=5, uniform=True, max_sntc_length=18, stop_on=None):
  seed_words = seed_words if torch.is_tensor(seed_words) else torch.tensor(seed_words)
  model.eval()
  with torch.no_grad():
    sampled_ix_list = seed_words.tolist()
    x = seed_words.unsqueeze(0).to(device)

    h,c = model.init_state(b_size=1)
    h, c = h.to(device), c.to(device)
    for t in range(max_sntc_length - len(seed_words)):
      sampled_ix, (h,c) = sample_function(model, x, (h,c), topk, uniform)
      h, c = h.to(device), c.to(device)
      sampled_ix_list.append(int(sampled_ix))
      x = torch.tensor([[sampled_ix]]).to(device)

      if sampled_ix==stop_on:
        break
  model.train()
  return sampled_ix_list



def train_with_BBTT(max_epochs, model, dataloader, criterion, optimizer, device, int_to_word, clip=None):
  losses = []
  perplexities = []
  sentences = []
  epoch = 0
  while epoch < max_epochs:
    epoch += 1
    running_loss = 0.0
    model.train()
    num_batches=0;
    for input, target in dataloader:
      num_batches = num_batches+1

      optimizer.zero_grad()
      input = input.to(device).to(torch.int64)
      target = target.to(device).to(torch.int64)

      h, c = model.init_state(b_size=input.shape[0])
      h = h.to(device)
      c = c.to(device)

      outputs, (h,c) = model(input, (h,c))
      outputs = outputs.transpose(1,2) # N VocabSize seq_len
      loss = criterion(outputs, target)
      running_loss = running_loss+loss.item()
      loss.backward()
      if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()

    loss_val = running_loss/num_batches
    perpl = np.exp(loss_val)
    running_loss = 0.0
    losses.append(loss_val)
    perplexities.append(perpl)
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'.format(epoch, max_epochs, loss_val, perpl))
    sentences.append(evaluate(model,dataloader,int_to_word,device))
  return model, losses, perplexities, sentences



def train_with_TBBTT(max_epochs, model, dataloader, criterion, optimizer, chunk_size, device, int_to_word, clip=None):
  losses = []
  perplexities = []
  sentences = []
  epoch = 0
  while epoch < max_epochs:
      epoch += 1
      running_loss = 0.0
      model.train()
      num_batches=0;
      for input, output in dataloader:
          n_chunks = input.shape[1] // chunk_size
          num_batches = num_batches+1

          for j in range(n_chunks):
              optimizer.zero_grad()

              if j < n_chunks - 1:
                  input_chunk = input[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
                  output_chunk = output[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
              else:
                  input_chunk = input[:, j * chunk_size:].to(device).to(torch.int64)
                  output_chunk = output[:, j * chunk_size:].to(device).to(torch.int64)

              if j == 0:
                  h, c = model.init_state(b_size=input.shape[0])
              else:
                  h, c = h.detach(), c.detach()
              h = h.to(device)
              c = c.to(device)
              outputs, (h,c) = model(input_chunk, (h,c))
              outputs = outputs.transpose(1,2) # N VocabSize seq_len
              loss = criterion(outputs, output_chunk)
              running_loss = running_loss+(loss.item()/n_chunks)
              loss.backward()
              if clip:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
              optimizer.step()

      loss_val = running_loss/num_batches
      perpl = np.exp(loss_val)
      running_loss = 0.0
      losses.append(loss_val)
      perplexities.append(perpl)
      print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'.format(epoch, max_epochs, loss_val, perpl))
      sentences.append(evaluate(model,dataloader,int_to_word,device))
  return model, losses, perplexities, sentences



def get_rand_partial_prompt(sentence):
  idx = np.random.choice(len(sentence)//2,1)[0]
  if(idx==0):
    idx=1
  return sentence[:idx]



def evaluate(model,dataloader,int_to_word, device, topk=5, uniform=None, max_sntc_length=30, input_sentence=None):
  def translate(indxs_list):
    avoid = [0,len(int_to_word)-1]
    res = []
    for i in indxs_list:
      if(i not in avoid):
        res.append(int_to_word[i])
    return res

  if input_sentence:
    x = input_sentence
    seed_words = input_sentence
  else:
    x,_ = next(iter(dataloader))
    x=x[0]
    seed_words = get_rand_partial_prompt(x).tolist()
    x=x.tolist()

  argmax_indxs = sample(model, seed_words, sample_argmax, device, topk=topk, uniform=uniform, max_sntc_length=max_sntc_length, stop_on=0)
  topk_indxs = sample(model, seed_words, random_sample_next, device, topk=topk, uniform=uniform, max_sntc_length=max_sntc_length, stop_on=0)
  o_sntc = " ".join(translate(x))
  seed_sntc = " ".join(translate(seed_words))
  arg_sntc = " ".join(translate(argmax_indxs))
  topk_sntc = " ".join(translate(topk_indxs))
  print("Original sentence -> "+o_sntc)
  print("Seed -> "+seed_sntc)
  print("Argmax prompt -> "+arg_sntc)
  print("Topk prompt -> "+topk_sntc+"\n")

  return [o_sntc, seed_sntc, arg_sntc, topk_sntc]


def checkpoint(model,optimizer,save_name=None):
  dic = {
      'model_state' : model.state_dict(),
      'optimizer' : optimizer.state_dict()
  }
  if save_name is not None:
    torch.save(dic, PATH+save_name+".pt")
  return dic

df = pd.read_json(PATH+'News_Category_Dataset_v3.json', lines=True)
print(df.info(),"\n")
print(df.shape,"\n")
print(df.describe,"\n")

df2 = df.loc[df['category'] == 'POLITICS']
print(df2.info(),"\n")
print(df.shape,"\n")
print(df2[0:3],"\n")

t = TreebankWordTokenizer()
tkns = [t.tokenize(hdline.lower())+["<EOS>"] for hdline in df2['headline']]
print(len(max(tkns, key=len)))

# uncomment the following block to save variables in pickle format
#save_to_pickle(tkns, PATH+'tokenized.pickle')
tkns = load_from_pickle(PATH+'tokenized.pickle')

all_words = []
frequency = {}
for t in tkns:
  for w in t:
    if w in frequency:
      frequency[w] = frequency[w]+1
    else:
      frequency[w] = 1
      all_words.append(w)


for i,f in enumerate(sorted(frequency, key=frequency.get, reverse=True)[:5]):
  print(f"{i+1}: {f} -> {frequency[f]} repetitions")

all_words.remove("<EOS>")
all_words = ["<EOS>"] + all_words + ["PAD"]
print(len(all_words))

int_to_word = {}
for i in range(len(all_words)):
  int_to_word.update({i:all_words[i]})
word_to_int ={}
for k,v in int_to_word.items():
  word_to_int.update({v:k})

print(len(word_to_int))

# uncomment the following block to save variables in pickle format
'''
save_to_pickle(all_words, PATH+'all_words.pickle')
save_to_pickle(int_to_word, PATH+'int_to_word.pickle')
save_to_pickle(word_to_int, PATH+'word_to_int.pickle')
'''

all_words =  load_from_pickle(PATH+'all_words.pickle')
int_to_word = load_from_pickle(PATH+'int_to_word.pickle')
word_to_int = load_from_pickle(PATH+'word_to_int.pickle')

print(int_to_word[len(int_to_word)-1])
dataset = Dataset(tkns,word_to_int)
print(dataset.__getitem__(2)[1].dtype)

batch_size = 64
collate = lambda batch: collate_fn(batch, pad_idx=word_to_int['PAD'])
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

vocab_size = len(all_words)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
embed_size = 64
hidden_size = 1024
num_layers = 2
max_epochs = 15
chunk_size = 10
learning_rate = 0.002

criterion = nn.CrossEntropyLoss()

model_standard = myNet(vocab_size, embed_size, hidden_size, num_layers,dropout=0.3).to(device)
optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=learning_rate)

model_standard, losses_standard, perplexities_standard, sentences_standard = train_with_BBTT(max_epochs, model_standard, dataloader, criterion, optimizer_standard, device, int_to_word, clip=1)

# uncomment the following block to save variables in pickle format
'''
checkpnt = checkpoint(model_standard,optimizer_standard,save_name="Fabian_Gobet_standard")

with open(PATH+'losses_standard.pickle', 'wb') as handle:
    pickle.dump(losses_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(PATH+'perplexities_standard.pickle', 'wb') as handle:
    pickle.dump(perplexities_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(PATH+'sentences_standard.pickle', 'wb') as handle:
    pickle.dump(sentences_standard, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

model_standard = myNet(vocab_size, embed_size, hidden_size, num_layers,dropout=0.3).to(device)
optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=learning_rate)

load_checkpnt_standard = torch.load("/content/drive/MyDrive/USI/DL/assignment3/Fabian_Gobet_standard.pt")
model_standard.load_state_dict(load_checkpnt_standard['model_state'])
optimizer_standard.load_state_dict(load_checkpnt_standard['optimizer'])

with open(PATH+'losses_standard.pickle', 'rb') as handle:
    losses_standard = pickle.load(handle)

with open(PATH+'perplexities_standard.pickle', 'rb') as handle:
    perplexities_standard = pickle.load(handle)

with open(PATH+'sentences_standard.pickle', 'rb') as handle:
    sentences_standard = pickle.load(handle)

model = myNet(vocab_size, embed_size, hidden_size, num_layers,dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model, losses, perplexities, sentences = train_with_TBBTT(max_epochs, model, dataloader, criterion, optimizer, chunk_size, device, int_to_word, clip=1)

# uncomment the following block to save variables in pickle format
'''
checkpnt = checkpoint(model,optimizer,save_name="Fabian_Gobet")

with open(PATH+'losses.pickle', 'wb') as handle:
    pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(PATH+'perplexities.pickle', 'wb') as handle:
    pickle.dump(perplexities, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(PATH+'sentences.pickle', 'wb') as handle:
    pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

model = myNet(vocab_size, embed_size, hidden_size, num_layers,dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

load_checkpnt = torch.load("/content/drive/MyDrive/USI/DL/assignment3/Fabian_Gobet.pt")
model.load_state_dict(load_checkpnt['model_state'])
optimizer.load_state_dict(load_checkpnt['optimizer'])

with open(PATH+'losses.pickle', 'rb') as handle:
    losses = pickle.load(handle)

with open(PATH+'perplexities.pickle', 'rb') as handle:
    perplexities = pickle.load(handle)

with open(PATH+'sentences.pickle', 'rb') as handle:
    sentences = pickle.load(handle)

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("Loss values during training")
#plt.plot(np.arange(1,len(losses_standard)+1,1), losses_standard, marker='o', linestyle='dashed', color='green', label='Standard-BBTT loss')
plt.plot(np.arange(1,len(losses)+1,1), losses, marker='o', linestyle='dashed', color='blue', label='TBBTT loss')
plt.plot([1,len(losses)], [1.5,1.5], marker='o', linestyle='dashed', color='orange', label='1.5 threshold')
plt.xlabel("epochs")
plt.ylabel("losses")

yticks = [losses[0]]
min_interval = (losses[0]-losses[-1])/10
for tick in losses:
  if yticks[-1]-tick > min_interval:
    yticks.append(tick)
plt.yticks([1.5]+yticks)
plt.xticks(np.arange(1,len(losses)+1,1))
plt.grid()
plt.legend()
plt.savefig(PATH+'losses_TBBTT_plot')
plt.show()

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("Perplexity values during training")
#plt.plot(np.arange(1,len(perplexities_standard)+1,1), perplexities_standard, marker='o', linestyle='dashed', color='orange', label='Standard-BBTT perpl.')
plt.plot(np.arange(1,len(perplexities)+1,1), perplexities, marker='o', linestyle='dashed', color='blue', label='TBBTT perpl.')
plt.xlabel("epochs")
plt.ylabel("perplexities")

yticks = [perplexities[0]]
min_interval = (perplexities[0]-perplexities[-1])/10
for tick in losses:
  if yticks[-1]-tick > min_interval:
    yticks.append(tick)
plt.yticks([1.5]+yticks+perplexities[1:6], fontsize=8)
plt.xticks(np.arange(1,len(losses)+1,1))
plt.grid()
plt.legend()
plt.savefig(PATH+'perplexities_TBBTT_plot')
plt.show()

sentences = ["americans are", "trump will", "a hero of"]
t = TreebankWordTokenizer()
input_sentences = []
for sntc in sentences:
  tokenized = t.tokenize(sntc.lower())+["<EOS>"]
  input_sentences.append([word_to_int[token] for token in tokenized])

for inpt in input_sentences:
  print("Standard-BBTT model")
  evaluate(model_standard,None,int_to_word,device, topk=5, uniform=None, max_sntc_length=30, input_sentence=inpt)
  print("TBBTT model")
  evaluate(model,None,int_to_word,device, topk=5, uniform=None, max_sntc_length=30, input_sentence=inpt)