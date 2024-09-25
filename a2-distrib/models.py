# models.py
import nltk 

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from nltk.util import ngrams
nltk.download('words') 
from nltk.corpus import words 
from nltk.metrics.distance  import edit_distance 
correct_words = words.words()


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

def spelling_correction(incorrect_words):
    corrected_words_list = []
    for i in range(0,len(incorrect_words)):
        words = incorrect_words[i] 
        word_vec_list = [] 
        for word in words: 
            temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]] 
            word_vec_list.append(temp)
        corrected_words_list.append(word_vec_list)
    return corrected_words_list

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self,model):
     #   super(NeuralSentimentClassifier, self).__init__()
        self.model = model 
       
    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]
        
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Change to predict 
        prediction = self.model.predict(ex_words, has_typos= bool)
        return prediction


# Create a Deep Averaging network model class
class DAN(nn.Module):
    def __init__(self, n_classes, n_hidden_units, vocab_size, emb_dim=300):
        super(DAN, self).__init__()
  #      self.classifier = NeuralSentimentClassifier()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.V = nn.Linear(emb_dim, n_hidden_units)
        self.g = nn.ReLU()
        self.W = nn.Linear(n_hidden_units, n_classes)
        self._softmax = nn.Softmax(dim=1)
  #      self.word_embedding=word_embedding
        
    def average(self, x):
        avg_emb = x.sum(axis=0, keepdims=True)
        avg_emb /=  len(x)
        return avg_emb
        
    def forward(self, x):
        avg = self.average(x)
        return self._softmax(self.W(self.g(self.V(avg))))

    def predict(self,  ex_words: List[str], has_typos: bool) -> int:            
        word_vec_list = []
        for word in ex_words: 
            word = word.lower() 
            word_vec = form_input(WordEmbeddings.get_embedding(word))
            word_vec_list.append(word_vec)
        x=torch.stack(word_vec_list)
        log_probs = self.forward(x)
        prediction = torch.argmax(log_probs,dim=1)
        return prediction


class DAN_Batch(nn.Module):
    def __init__(self, n_classes, n_hidden_units, vocab_size, emb_dim=300,batch_size=34):
        super(DAN_Batch, self).__init__()
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.V = nn.Linear(emb_dim_t, n_hidden_units_t)
        self.g = nn.ReLU()
        self.W = nn.Linear(n_hidden_units_t, n_classes_t)

        self.log_softmax = nn.LogSoftmax(dim=None)
        self._softmax = nn.Softmax(dim=None)

        
    def average(self, x, z):
        q=torch.sum(x,(1), keepdims=True)
        avg_emb = q.view(len(z), -1)
        a = z.unsqueeze(1)
        a = a.type('torch.LongTensor')
        avg_emb = avg_emb /  a
        return avg_emb
        
    def forward(self, x):
        avg = self.average(x,z)
#        return self.log_softmax(self.classifier(avg))
        return self.log_softmax(self.W(self.g(self.V(avg))))
#        return self._softmax(self.W(self.g(self.V(avg))))        

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = document_batches['text']
        self.y = document_batches['labels']
        self.len = self.x.shape[0]
        self.z = document_batches['len']
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.z[index]
    
    # Getting length of the data
    def __len__(self):
        return self.len
  
def create_batch(batch):
    sentence_len = list()
    label_list = list()
    for ex in batch:
        sentence_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(sentence_len),max(sentence_len), batch[0][0].size(1)).zero_()
    for i in range(len(sentence_len)):
        sentence_text = batch[i][0]
        sentence_text_pad = F.pad(input=sentence_text, pad=(0, 0, max(sentence_len)-len(sentence_text), 0), mode='constant', value=0)
        x1[i, :].copy_(sentence_text_pad)
    sent_batch = {'text': x1, 'len': torch.FloatTensor(sentence_len), 'labels': target_labels}
    return sent_batch


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    
    labels_dev = []
    sentence_dev = []
    for i in range(0,len(dev_exs)):
        labels_dev.append(dev_exs[i].label)
        sentence_dev.append(dev_exs[i].words)
    
    labels = []
    sentence = []
    for i in range(0,len(train_exs)):
        labels.append(train_exs[i].label)
        sentence.append(train_exs[i].words)
    
    words_list = []
    
    for inner_list in sentence:
        words_list.extend(inner_list)
    vocab = set(words_list)
    vocab_size=len(vocab)
    document = []
    for i in range(0,len(train_exs)):
        target = labels[i]
        words = sentence[i] 
        word_vec_list = [] 
        for word in words: 
            word = word.lower() 
            word_vec = form_input(word_embeddings.get_embedding(word))
            word_vec_list.append(word_vec)
        document.append((torch.stack(word_vec_list), torch.tensor(target)))

    #document_batches=create_batch(document)
    # Define some constants
    embedding_size = word_embeddings.vectors[0].shape[0] 
    num_classes = 2
    num_hidden_units = 10
    dan = DAN(n_classes = num_classes, n_hidden_units = num_hidden_units , vocab_size = vocab_size,emb_dim=embedding_size)
    initial_learning_rate=0.001
    optimizer = optim.Adam(dan.parameters(), lr=initial_learning_rate)
    
    num_epochs = 4
    criterion = nn.CrossEntropyLoss()
    dan.train()
    for epoch in range(0, num_epochs):
        total_loss = 0.0
        for sentence,label in document:
            x = sentence
            y = label 
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            dan.zero_grad()
            log_probs = dan.forward(x)
            loss = criterion(log_probs.view(num_classes),y_onehot) 
            total_loss += loss 
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    dan.eval()
    classifier = NeuralSentimentClassifier(dan)
    return classifier

