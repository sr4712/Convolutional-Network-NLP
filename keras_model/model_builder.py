#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import time
from text_processor.text_processor import KerasTextListPreparer
from keras_model.keras_model import ConvolutionModel,WordEmbedder


class ModelBuilder:
    
    '''Class to implement model building
    
    
    # Arguments:
        conf: dictionary of params
        text_processor: text processor object
        model: model object
        word_embedder: word embedder object    
    '''
    
    def __init__(self,conf,text_processor,model,word_embedder):
        
        self.corpus_filename = conf['corpus_filename']
        self.labels_filename = conf['labels_filename']
        self.model_params = conf['model_params']        
        self.text_proc_params = conf['text_proc_params']
        self.word_embedding_params = conf['word_embedding_params']      
        self.text_list = self.load_corpus()     
        self.labels = self.load_labels()
        self.text_processor = text_processor(self.text_list,                          
                            self.text_proc_params['sentence_start_token'],
                            self.text_proc_params['sentence_end_token'],
                            self.text_proc_params['unknown_token'],
                            self.text_proc_params['pad_token'],
                            self.model_params['vocab_size'], 
                            self.model_params['num_timesteps'],
                            self.labels)
        self.x_train_all_samples,self.y_train_all_samples  =  self.\
                                                        text_processor.create_padded_training_data()   
               
        if self.model_params['model_type'] in ['CNN-static','CNN-non-static']:            
            self.word_embedder_obj = word_embedder(
                                                 self.word_embedding_params['embedding_directory'], 
                                                 self.word_embedding_params['embedding_file'],
                                                 self.model_params['vocab_size'],
                                                 self.model_params['embed_dim'],
                                                 self.text_proc_params['sentence_start_token'],
                                                 self.text_proc_params['sentence_end_token'],
                                                 self.text_proc_params['unknown_token'],
                                                 self.text_processor.word_to_index_dict)           
            self.pretrained_word_embeddings_matrix = self.word_embedder_obj.create_embedding_matrix()
            self.model_params['embedding_matrix'] = self.pretrained_word_embeddings_matrix     
            
        self.conv_model = model(
                              self.x_train_all_samples,
                              self.y_train_all_samples,
                              ** self.model_params )            
                           
        self.trained_conv_model = self.train_model()
               
    def load_corpus(self):
        try:
            pkl_file = open(self.corpus_filename,'rb')
            text_list = pickle.load(pkl_file)
            pkl_file.close()
        except FileNotFoundError:
            print('Corpus file was not found.')
        return(text_list)  
    
    def load_labels(self):
        try:
            pkl_file = open(self.labels_filename,'rb')
            text_list = pickle.load(pkl_file)
            pkl_file.close()
        except FileNotFoundError:
            print('Labels file was not found.')
        return(text_list) 
    
    def train_model(self):
        model_history = self.conv_model.fit_model()
        return(model_history)
    
    @staticmethod
    def create_filename(filename_input):
        current_date  =  time.strftime("%d_%m_%Y")
        filename_output = current_date+'_'+filename_input
        return(filename_output)

    def output_model_description(self):
        self.conv_model.output_model_summary()
        self.conv_model.output_model_layer_description()

    def save_training_data(self):
        print('saving training data')
        filename_x = time.strftime("%d_%m_%Y")+'_'+'x_train_all_samples.npy'
        filename_y = time.strftime("%d_%m_%Y")+'_'+'y_train_all_samples.npy'
        np.save(filename_x,self.x_train_all_samples)
        np.save(filename_y,self.y_train_all_samples)
        
    def save_word_to_index_dict(self):
        print('saving word_to_index_dict')
        filename_dict = time.strftime("%d_%m_%Y")+'_'+'word_to_index_dict.pkl'
        with open(filename_dict, 'wb') as handle:
            pickle.dump(
                        self.text_processor.word_to_index_dict, handle, pickle.HIGHEST_PROTOCOL)
    
    def save_embedding_matrix(self):
        if self.model_params['model_type'] in ['CNN-static','CNN-non-static']:
            print('saving embedding matrix...')
            filename_embed_matrix = self.create_filename('embed_matrix.pickle')
            with open(filename_embed_matrix,'wb') as handle:
                pickle.dump(
                            self.model_params['embedding_matrix'],
                            handle,
                            protocol = pickle.HIGHEST_PROTOCOL)
                    
    def save_model(self):
        print('saving model...')
        filename_model = self.create_filename('text_generation_model.h5')
        self.conv_model.training_model.save(filename_model)

    def save_model_results(self):
        print('saving model results...')
        model_results_dict = self.trained_conv_model.history       
        filename_results = self.create_filename('model_results.pickle')
        with open(filename_results, 'wb') as handle:
            pickle.dump(model_results_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def save_model_and_session_info(self):
        self.save_training_data()
        self.save_word_to_index_dict()
        self.save_embedding_matrix()
        self.save_model()
        self.save_model_results()
        
    def save_model_info(self):
        self.save_model()
        self.save_model_results()


if __name__=='__main__':    
    conf = {'corpus_filename':'required_objects/sa_labelled_sentences.pickle',
            'labels_filename':'required_objects/sentence_labels.pickle',
          'text_proc_params':{
                  'sentence_start_token':'SENTENCE_START',
                  'sentence_end_token':'SENTENCE_END',
                  'unknown_token':'UNKNOWN_TOKEN',
                  'pad_token':'PAD_TOKEN'
                  },
          'word_embedding_params':{
              'embedding_directory':'/Volumes/GLOVE/glove.6B',
              'embedding_file':'glove.6B.100d.txt'
              },
          'model_params':{
                  'vocab_size':1000,
                  'num_timesteps':8,
                  'embed_dim':100,
                  'model_type':'CNN-non-static'
                  }}
    
    model_builder = ModelBuilder(conf,KerasTextListPreparer,ConvolutionModel,WordEmbedder)
    model_builder.output_model_description()
    model_builder.save_model_and_session_info()