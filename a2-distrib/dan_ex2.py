#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:24:21 2024

@author: erichogue
"""
def __init__(self, n_classes,vocab_size, emb_dim=300,n_hidden_units =300):
    super(DAN,self).__init__
    self.n_classes = n_classes
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.n_hidden_units = n_hidden_units
    self.embeddings = nn.Embedding(self.vocab_size,self.emb_dim)
    self.classifier = nn.Sequential(nn.Linear(self.n_hidden_units,self.n_hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(self.n_hidden_units,
                                              self.n_classes))
    self._softmax = nn.Softmax()


def forward(self,batch,probs=False):
    text = batch['text']['tokens']
    length = batch['length']
    text_embed = self._word_embeddings(text)
    encoded = text_embed.sum(1)
    encoded /= lenghts.view(text_embed.size(0),-1)
    logits = self.classifier(encoded)
    if probs:
        return self._softmax(logits)
    else: 
        return logits 

def _run_epoch(self,batch_iter,train=True):
    self._model.train()
    for batch in batch_iter:
        model.zero_grad()
        out = model(batches)
        batch_loss = criterion(out,batch['lable'])
        batch_loss.backward()
        self.optimizer.step() 

This function takes in multiple inputs, stored in one tensor x. Each input is a bag of word 
representation of reviews. For each review, it retrieves the word embedding of each word in 
the review and averages them (weighted by the corresponding
        entry in x). 