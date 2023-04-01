import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import re
from io import open
import torch
from torch import nn, optim 
import random 



class LSTM_Encoder(nn.Module):
    """
    define a LSTM encoder that takes a token and produces a final state
    _, final_state = your_lstm_encoder(token, ...)
    """
    def __init__(self, input_size: int, embed_size: int, hidden_size: int, 
                 num_layers: int=1, bidirectional: bool=False, dropout: float=0):
      """
      input_size: vocabulary size
      embed_size: embedding size
      hidden_size: LSTM size
      """
      super(LSTM_Encoder, self).__init__()

      self.embedding = nn.Embedding(input_size, embed_size)
        
      self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bias=True, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional, proj_size=0)
                
    def forward(self, src: torch.LongTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
      """ 
      compute context for input batch

      @param:
      src: source sequence [batch_size, seq_length]

      return:
      hidden_state: [num_layer * n_direction, batch_size, hidden_size]
      cell_state: [num_layer * n_direction, batch_size, hidden_size]
      """
      # batch shape [10, 50])
      embed = self.embedding(src) # [10, 50, 100]

      # [10, 50, 64], [1, 10, 64], [1, 10, 64]
      output, (hidden_state, cell_state) = self.rnn(embed)
      
      # [1, 10, 64], [1, 10, 64]
      return hidden_state, cell_state
 



class LSTM_Decoder(nn.Module):
    """
    a LSTM decoder that takes the final state from previous step and produces a sequence of outputs
    outs, _ = your_lstm_decoder(final_state, ...)
    """
    def __init__(self, output_dim: int, embed_size: int, hidden_size: int,
                 num_layers: int=1, bidirectional: bool=False, dropout: float=0):
        """
        output_dim: vocabulary size 
        """
        super(LSTM_Decoder, self).__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, embed_size)
        
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bias=True, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional, proj_size=0)
        self.linear = nn.Linear(hidden_size, output_dim)
                
    def forward(self, input: torch.LongTensor, hidden: torch.FloatTensor, cell: torch.FloatTensor) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        predict the corresponding tokens in a batch 

        @parmas:
        input: token [batch_size, 1]  (1 is sequence length)
        hidden: hidden states of previous input. shape [num_layer * n_direction, batch_size, hidden_size] 
        cell: cell states of previous input. shape [num_layer * n_direction, batch_size, hidden_size]

        return:
        prediction: 3d torch.LongTensor. shape [batch_size, 1, vocab_size]
        hidden_state: 3d torch.FloatTensor. shape [num_layer * n_direction, batch_size, hidden_size]
        cell_state: 3d torch.FloatTensor. shape [num_layer * n_direction, batch_size, hidden_size]
        """
        # embed: [10, 50] batch: [10, 1]
        embed = self.embedding(input) 

        # [10, 1, 64], [1, 10, 64], [1, 10, 64]
        output, (hidden_state, cell_state) = self.rnn(embed, (hidden, cell))

        prediction = self.linear(output)
        
        # [10, 1, 6654], [1, 10, 64], [1, 10, 64]
        return prediction, hidden_state, cell_state



class Seq2Seq(nn.Module):
  """a LSTM autoencoder with 2 components: a LSTM encoder and a LSTM decoder
     Unsupervised learning task: given a sequence of tokens, predict next token 
  """
  def __init__(self, encoder: LSTM_Encoder, decoder: LSTM_Decoder, device):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder 
    self.device = device 
  
  def forward(self, src: torch.LongTensor, teacher_forcing_ratio: float=0) -> torch.LongTensor:
    """
    predict the corresponding tokens in a batch 

    @param:
    src: source sequence. shape: [batch_size, seq_length]
    teacher_forcing_ratio: if higher than this ratio, use ground-truth next token as input to decoder;
      otherwise, use predicted next token as input to decoder

    return:
    outputs: predicted tokens [batch_size, seq_length, vocab_size]
    """
    batch_size, seq_length = src.shape
    VOCAB_SIZE = self.decoder.output_size

    hidden, cell = self.encoder(src)

    # initialize outputs [batch_size, seq_length, vocab_size]
    outputs = torch.zeros(batch_size, seq_length, VOCAB_SIZE).to(self.device)

    # [batch_size, 1]
    input = torch.zeros(batch_size, 1, dtype=torch.int64)  # first input is BOS token
    # input = src[:,0].view(batch_size, -1)   # first input is first token of source sequence

    for i in range(1, seq_length):
      output, hidden, cell = self.decoder(input, hidden, cell)

      outputs[:,i,:] = output.view(batch_size, -1)  

      # teacher forcing
      if random.random() < teacher_forcing_ratio:
        input = src[:,i].view(batch_size, -1)    # [batch_size, 1]
      else: 
        input = output.view(batch_size, -1).argmax(1, keepdim=True)   # [batch_size, 1]

    return outputs



def train(device, model, optimizer, loss_fn, tokens, vocabulary, **kwargs):

  iteration, batch_size, seq_length, vocab_size = kwargs['iteration'], kwargs['batch_size'], kwargs['seq_length'], kwargs['vocab_size']
  model.train()
  losses = []
  i = 0
  for num_iter in range(iteration):
      
      # create batch data 
      batch = [[vocabulary.index(v) for v in tokens[ii:ii + seq_length]] for ii in range(i, i + batch_size)]
      batch = np.stack(batch, axis=0)
      batch = torch.tensor(batch, dtype=torch.long)
      batch = batch.to(device) 
      i += batch_size
      if i + batch_size + seq_length > len(tokens): i = 0

      optimizer.zero_grad()

      out = model(batch, teacher_forcing_ratio=kwargs['teacher_forcing_ratio']).view(batch_size, VOCAB_SIZE, seq_length)

      loss = loss_fn(out, batch)
      loss.backward()
      optimizer.step()
      
      losses.append(loss.item())

      if num_iter % 100 == 0: 
          print(f'iteration: {num_iter} \tTraining Loss: {loss.item():.3f}')

  torch.save(model.state_dict(), './checkpoint.pt')

  with open('./loss.npy', 'wb') as f:
    np.save(f, losses)

  return model 




def plot_word_embeddings(device, model, vocabulary, vocab_size):
  """Visualize word embeddings in 2D by PCA"""
  model.eval()
  vocab = torch.tensor(range(vocab_size)).to(device)
  embeds = model.encoder.embedding(vocab).detach().cpu().numpy()
  # shape: (6654, 2)
  embeds_pca = sklearn.decomposition.PCA(2).fit_transform(embeds)

  # plot sampled word embeddings, run 5 times
  for i in range(5):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(embeds_pca[:, 0], embeds_pca[:, 1], s=5, c='w')
    MIN_SEPARATION = .1 * min(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])

    xy_plotted = set()

    for i in np.random.choice(vocab_size, vocab_size, replace=False):
        x_, y_ = embeds_pca[i]
        if any([(x_ - point[0])**2 + (y_ - point[1])**2 < MIN_SEPARATION for point in xy_plotted]): continue
        xy_plotted.add(tuple([embeds_pca[i, 0], embeds_pca[i, 1]]))
        ax.annotate(vocabulary[i], xy=embeds_pca[i])
  

def main():

  #############################################
  # load data, create vocabulary
  #############################################
  ! wget --no-check-certificate --content-disposition "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt" --output-document="./data/J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt"
  TEXT_FILE = "./data/Harry Potter 1 - Sorcerer's Stone Chapter 1.txt"
  string = open(TEXT_FILE).read()
  # convert text into tokens
  tokens = re.split('\W+', string)
  # create vocabulary
  vocabulary = sorted(set(tokens))
  vocab_size = len(vocabulary)
  print(f'vocab size: {vocab_size}')


  #############################################
  # hyperparameters
  #############################################
  # training params
  train_params = {
      'iteration': 1000,
      'lr': 0.001,
      'batch_size': 128}

  # model params
  model_params = {
  'seq_length': 10,
  'vocab_size': vocab_size,
  'embed_size': 256,
  'hidden_size': 256,
  'num_layers': 2, # for both encoder and decoder
  'dropout': 0.5, # for both encoder and decoder
  'teacher_forcing_ratio': 0.5,
  'bidirectional': False} 

  kwargs = {**train_params, **model_params}

  #############################################
  # initialize model, optimizer, loss function
  #############################################
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  encoder = LSTM_Encoder(model_params['vocab_size'], model_params['embed_size'], model_params['hidden_size'], model_params['num_layers'], model_params['bidirectional'], model_params['dropout']).to(device)
  decoder = LSTM_Decoder(model_params['vocab_size'], model_params['embed_size'], model_params['hidden_size'], model_params['num_layers'], model_params['bidirectional'], model_params['dropout']).to(device)
  model = Seq2Seq(encoder, decoder).to(device)

  optimizer = optim.Adam(model.parameters(),lr=kwargs['lr'])
  loss_fn = nn.CrossEntropyLoss()  # input (batch_size, num_class) target (batch_size)

  #############################################
  # training
  #############################################
  model = train(device, model, optimizer, loss_fn, tokens, vocabulary, **kwargs)

  #############################################
  # evaluation
  #############################################
  plot_word_embeddings(device, model, vocabulary, model_params['vocab_size'])

if __name__ == '__main__':
   main()
