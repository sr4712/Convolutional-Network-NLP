#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import nltk
import re
import string
import numpy as np
from keras.preprocessing import sequence


class TextListCleaner:  
     
    '''Cleans text contained in a list, 
    where every element in the list is a string
    
    
    # Arguments:
        text_list: a list containing text 
        
    '''
    
    def __init__(self,text_list):
        self.text_list = text_list    

    def copy_list(self):
        text_list_copy = list(self.text_list)
        return(text_list_copy)
    
    @staticmethod
    def lowercase_text(text_list):
        text_list = [x.lower() for x in text_list]
        return(text_list)
    
    @staticmethod
    def remove_punctuation_from_text(text_list,punct_to_keep):
        punct_chars_to_delete = set(string.punctuation) - {punct_to_keep}
        text_list = [''.join(c for c in s if c not in punct_chars_to_delete) for s in text_list]
        return(text_list)
    
    @staticmethod
    def remove_tab_from_text(text_list):
        text_list = [re.sub('\t', '', x) for x in text_list]
        return(text_list)
    
    @staticmethod
    def remove_newline_from_text(text_list):
        text_list = [re.sub('\n', '', x) for x in text_list]
        return(text_list)
    
    @staticmethod
    def remove_extra_spaces_from_text(text_list):
        text_list = [' '.join(x.split()) for x in text_list]
        return(text_list)
    
    def clean_text(self):
        text_list_clean = self.copy_list()
        text_list_clean = self.lowercase_text(text_list_clean)
        text_list_clean = self.remove_punctuation_from_text(text_list_clean,'')
        text_list_clean = self.remove_tab_from_text(text_list_clean)
        text_list_clean = self.remove_newline_from_text(text_list_clean)
        text_list_clean = self.remove_extra_spaces_from_text(text_list_clean)
        return(text_list_clean)
    
    def tokenize_clean_string_into_sentences(self):
        '''Method acceps a list of strings as input and 
        converts to a list of sentences
        '''
        sentence_list = list([nltk.sent_tokenize(x) for x in self.clean_text()])
        sentence_list = list(itertools.chain(*sentence_list))
        return(sentence_list)
    

class KerasTextListPreparer:
    
    '''Prepare text as a list of strings for keras
    
    
    # Arguments:
        text_list: a list containing text
        sentence_start_token: a token for starting sentences
        sentence_end_token: a token for ending sentences
        unknown_token: a token for unknown_words
        pad_token: a token for the padding input
        vocab_size: vocab_size for keras model
        num_timesteps: numer of timesteps in a sequence 
        classification_labels: classification labels for sentence
    
    '''
    
    def __init__(
            self,
            text_list,
            sentence_start_token,
            sentence_end_token,
            unknown_token,
            pad_token,
            vocab_size,
            num_timesteps,
            classification_labels = None):
        
        self.text_cleaner = TextListCleaner(text_list)
        self.classification_labels = classification_labels
        self.vocab_size = vocab_size
        self.sentence_start_token = sentence_start_token
        self.sentence_end_token = sentence_end_token
        self.unknown_token = unknown_token 
        self.pad_token = pad_token
        self.num_timesteps = num_timesteps        
        self.sentence_list = None 
        self.tokenize_sentences = None
        self.vocab = None
        self.word_to_index_dict = None
        self.index_to_word_dict = None
        self.sentence_list = self.create_sentence_list()
        self.tokenize_sentences = self.tokenize_sentences_to_words()
        self.vocab = self.create_vocab()
        self.word_to_index_dict = self.create_word_to_index_dict()
        self.index_to_word_dict = self.create_index_to_word_dict()
        
    def create_sentence_list(self):
        sentence_list = self.text_cleaner.tokenize_clean_string_into_sentences()
        sentence_list = self.text_cleaner.remove_punctuation_from_text(sentence_list,'')
        sentence_list = [
                "%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) 
                for x in sentence_list
                ]
        return(sentence_list)

    def tokenize_sentences_to_words(self):
        if self.sentence_list :
            sentence_tokenized_to_word_list = [nltk.word_tokenize(x) for x in self.sentence_list]
        else:
            raise Exception('self.sentence_list has not been constructed and is NoneType')
        return(sentence_tokenized_to_word_list)
    
    def create_word_freq_dist(self):
        if self.tokenize_sentences:
            word_freq_dist = nltk.FreqDist(itertools.chain(*self.tokenize_sentences))
        else:
            raise Exception('self.tokenize_sentences has not been constructed and is NoneType')
        return(word_freq_dist)
    
    def calculate_num_unique_words_in_corpus(self):
        word_freq_dist = self.create_word_freq_dist()
        return(len(word_freq_dist.items()))
    
    def create_vocab(self):
        word_freq_dist = self.create_word_freq_dist()
        num_unique_words = self.calculate_num_unique_words_in_corpus()
        if num_unique_words <self.vocab_size:
            raise Exception('vocab_size is less than the total unique word tokens and is NoneType')
        vocab = word_freq_dist.most_common(self.vocab_size-1)                                   
        return(vocab)
    
    def create_word_to_index_dict(self):
        if self.vocab:
            index_to_word = [x[0] for x in self.vocab]                     
            index_to_word.append(self.unknown_token)
            word_to_index_dict = dict([(w,i) for i,w in enumerate(index_to_word,1)])  
            word_to_index_dict[self.pad_token] = 0                        
        else:
            raise Exception('self.vocab has not been constructed and is NoneType')
        return(word_to_index_dict)

    def create_index_to_word_dict(self):
        if self.word_to_index_dict:
            index_to_word_dict = dict((v,k) for k,v in self.word_to_index_dict.items())                        
        else:
            raise Exception('self.word_to_index_dict has not been constructed and is NoneType.')
        return(index_to_word_dict)

    def create_training_sentences(self):
        if self.tokenize_sentences or self.word_to_index_dict:
            training_sentences = []
            for i, sent in enumerate(self.tokenize_sentences):
                training_sentences.insert(i, [
                                              w if w in self.word_to_index_dict 
                                              else self.unknown_token for w in sent
                                              ])
        else:
            raise Exception('self.tokenize_sentences or self.word_to_index_dict are NoneType')
        return training_sentences    
    
    
    def create_training_data(self):
        training_sentences = self.create_training_sentences()        
        x_train_all_samples = np.asarray([[
                                           self.word_to_index_dict[w] for w in sent] 
                                           for sent in training_sentences
                                           ])
        y_train_all_samples = np.asarray(self.classification_labels)
        return(x_train_all_samples,y_train_all_samples)
    
    def create_padded_training_data(self):        
        x_train_all_samples, y_train_all_samples=self.create_training_data()
        x_train_all_samples = sequence.pad_sequences(
                                x_train_all_samples, maxlen = self.num_timesteps, padding = 'post',
                                truncating = 'post', value=0)           
        return(x_train_all_samples,y_train_all_samples)

