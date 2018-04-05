#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod
import numpy as np
import os
from keras import optimizers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM, TimeDistributed
from keras.layers.merge import Concatenate
    

class LanguageModel(metaclass = ABCMeta):
    
    ''' Abstract Base Class to help build keras Language Model for classification tasks
    
    
    # Arguments:
        x: numpy array containig training data
        y: numpy array containing labels
        vocab_size: vocab_size for keras model
        num_timesteps: number of timesteps in model
        batch_size: number of samples for every gradient update (int)
        num_epochs: number of epochs to use for training (int)
        validation_split: fraction of data that should be reserved
                          for validating model when building it 
                          (float between 0 and 1)
        optimizer: optimizer to use when minizing loss function
        loss: type of loss to use when building model
        metrics: metric to use when building model        
        '''
            
    def __init__(
                self,
                 x,
                 y,
                 vocab_size,
                 num_timesteps,
                 batch_size = 128,
                 num_epochs = 2,
                 validation_split = .2,
                 learning_rate = .01,
                 learning_rate_decay = 0,
                 optimizer_type = 'Adagrad',
                 loss = 'binary_crossentropy',
                 metrics = 'accuracy'):
        
        self.x = x
        self.y = y
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.loss = loss
        self.metrics = metrics
        self.training_model = self.get_model()
        self.optimizer = self.create_optimizer()

    @abstractmethod
    def build_model_architecture(self):
        return

    def get_model(self):
        self.training_model = self.build_model_architecture()
        return self.training_model

    def create_optimizer(self):
        if self.optimizer_type == 'Adagrad':
            self.optimizer = optimizers.Adagrad(lr = self.learning_rate, decay = self.learning_rate_decay)
        if self.optimizer_type == 'RMSprop':
            self.optimizer = optimizers.RMSprop(lr = self.learning_rate, decay = self.learning_rate_decay)        
        if self.optimizer_type == 'Adadelta':
            self.optimizer = optimizers.Adadelta(lr = self.learning_rate, decay = self.learning_rate_decay)
        return self.optimizer  
    
    def compile_model(self):
        self.training_model.compile(loss = self.loss,
                                    optimizer = self.optimizer, 
                                    metrics = [self.metrics])
        
    def copy_training_data(self):
        return(self.x.copy(),self.y.copy())

    def shuffle_data(self,data_input):
        np.random.seed(421)
        num_values = data_input.shape[0]
        val_indices = np.arange(num_values)
        np.random.shuffle(val_indices)
        return(data_input[val_indices])

    def create_data_validation_split(self):
        x_train_all_samples,y_train_all_samples = self.copy_training_data()
        x_train_all_samples = self.shuffle_data(x_train_all_samples)
        y_train_all_samples = self.shuffle_data(y_train_all_samples)
        num_validation_samples = int(self.validation_split * x_train_all_samples.shape[0])
        x_train = x_train_all_samples[:-num_validation_samples]
        y_train = y_train_all_samples[:-num_validation_samples]
        x_val = x_train_all_samples[-num_validation_samples:]
        y_val = y_train_all_samples[-num_validation_samples:]
        return(x_train,y_train,x_val,y_val)

    def fit_model(self):
        self.compile_model()                                                          
        x_train,y_train,x_val,y_val = self.create_data_validation_split()
        training_model_history = self.training_model.fit(
                                                    x_train, 
                                                    y_train,
                                                    batch_size = self.batch_size,
                                                    epochs = self.num_epochs,
                                                    validation_data = (x_val, y_val))
        return(training_model_history)

    def output_model_summary(self):
        print('model summary',self.training_model.summary())

    def output_model_layer_description(self):
        layer_names_list = [self.training_model.layers[x].name 
                          for x in range(len(self.training_model.layers))
                          ]   
        layer_descriptions_list = [[str(e) for e in self.training_model.layers[x].trainable_weights] 
                                 for x in range(len(self.training_model.layers))]   
        layer_description = list(zip(layer_names_list,layer_descriptions_list))
        print('Layer Descriptions',layer_description)

    @abstractmethod
    def predict_with_model(self,input_sentence):
        self.training_model.predict(input_sentence)


class ConvolutionModel(LanguageModel):                                    
    
    '''Class to build keras Convolution Model with word embeddings for classification tasks
    
    
    # Arguments:
        embed_dim: embedding dimension for word vectors 
        mask_zero: whether to mask a value (default 0) when training (T,F)
        trainable: whether embedding matrix should be trainingable (T,F)
        lstm_hidden_dim: dimension of hidden layer for LSTM unit
        return_sequences: whether to return the full sequences (T,F)
        activation: activation functio nto use in output layer   
    '''

    def __init__(self, *args, **kwargs):    
        self.embed_dim = kwargs.pop('embed_dim')
        self.model_type = kwargs.pop('model_type','CNN-non-static')
        self.filter_sizes = kwargs.pop('filter_sizes',[3,5])
        self.num_filters = kwargs.pop('num_filters',128)
        self.pool_size = kwargs.pop('pool_size',2)
        self.dropout_prob_for_embed_layer = kwargs.pop('dropout_prob_for_embed_layer',.5)
        self.dropout_prob_for_concat_layer = kwargs.pop('dropout_prob_for_concat_layer',.5)
        self.dropout_prob_for_hidden_layer = kwargs.pop('dropout_prob_for_hidden_layer',.5)
        self.hidden_dims = kwargs.pop('hidden_dims',50)
        self.dense_layer_regularization = kwargs.pop('dense_layer_regularization',.5)
        self.conv_regularization = kwargs.pop('conv_regularization',0)
        self.embedding_matrix = kwargs.pop('embedding_matrix',None)
        super(ConvolutionModel, self).__init__(*args, **kwargs)


    def check_pretrained_word_embeddings_inputs_match(self):
        if (self.model_type in ['CNN-static','CNN-non-static']) and self.embedding_matrix is None:
            raise ValueError(
                      'Pretrained work embeddings are required, but no embedding_matrix'\
                    + 'has been provided to constructor')
        if self.model_type == 'CNN-rand' and self.embedding_matrix is not None:
            raise ValueError(
                        'Pretrained word embeddings are not required, but an embedding_matrix'\
                      + 'has been provided to constructor')   
            
    def build_model_architecture(self):
        self.check_pretrained_word_embeddings_inputs_match()
        input_shape = (self.num_timesteps,)
        conv_model_input = Input(shape=input_shape)
        
        if self.model_type == 'CNN-static':
            model_build_step = Embedding(self.vocab_size+1, self.embed_dim, input_length=self.num_timesteps, weights=[self.embedding_matrix],trainable=False,name="embedding")(conv_model_input)    
        elif self.model_type == 'CNN-non-static':
            model_build_step = Embedding(self.vocab_size+1, self.embed_dim, input_length=self.num_timesteps, weights=[self.embedding_matrix],trainable=True,name="embedding")(conv_model_input)
        elif self.model_type == 'CNN-rand':
            model_build_step = Embedding(self.vocab_size+1, self.embed_dim, input_length=self.num_timesteps, weights=None,trainable=True, name="embedding")(conv_model_input)

        Dropout(self.dropout_prob_for_embed_layer)(model_build_step)
 
        conv_blocks = []
        for filter_size_iter in self.filter_sizes:
            conv = Convolution1D(filters=self.num_filters,
                                 kernel_size=filter_size_iter,
                                 padding="valid",
                                 activation="relu",
                                 kernel_regularizer = regularizers.l2(self.conv_regularization),
                                 strides=1)(model_build_step)
            conv = MaxPooling1D(pool_size=self.pool_size)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        model_build_step = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]


        model_build_step = Dropout(self.dropout_prob_for_concat_layer)(model_build_step)
        model_build_step = Dense(self.hidden_dims,kernel_regularizer = regularizers.l2(self.dense_layer_regularization), activation="relu")(model_build_step)

        model_build_step = Dropout(self.dropout_prob_for_hidden_layer)(model_build_step)
        conv_model_output = Dense(1, activation="sigmoid")(model_build_step)

        conv_model = Model(conv_model_input, conv_model_output)
        return conv_model

    def predict_with_model(self):
        print('in subclass predict method')


class WordEmbedder:
    
    '''Class to process word embeddings
    
    
    # Arguments:
        embedding_directory: directory where pretrained embeddings are located 
        embedding_file: name of embedding file
        vocab_size: vocab size for keras model
        embed_dim: dimension for word vectors
        sentence_start_token: token to indicate start of sentence
        sentence_end_token: token to indicate end of sentence
        unknown_token: token for unknown words
        word_to_index_dict: dict the maps words to numerical indices
    ''' 
    
    def __init__(
                self,
                embedding_directory,
                embedding_file,
                vocab_size,
                embed_dim,
                sentence_start_token,
                sentence_end_token,
                unknown_token,
                word_to_index_dict):
        
        self.embedding_directory = embedding_directory
        self.embedding_file = embedding_file
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sentence_start_token = sentence_start_token
        self.sentence_end_token = sentence_end_token
        self.unknown_token = unknown_token
        self.word_to_index_dict = word_to_index_dict
            
    def create_word_to_embeddings_dict(self):
        word_to_embeddings_dict = {}
        file_embed = open(os.path.join(self.embedding_directory,self.embedding_file))
        for line in file_embed:
            computed_embeddings = line.split()
            words = computed_embeddings[0]
            word_vectors = np.asarray(computed_embeddings[1:],dtype = 'float32')
            word_to_embeddings_dict[words] = word_vectors
        file_embed.close()
        return word_to_embeddings_dict
    
    def check_word_embedding_dimension_correct(self,word_to_test):
        word_to_embeddings_dict = self.create_word_to_embeddings_dict()
        test_vec = word_to_embeddings_dict[word_to_test]
        if test_vec.shape[0] != self.embed_dim:
            raise ValueError('Embedding vectors do not have the correct embedding dimension!')
    
    def create_init_vectors_for_tokens(self,random_seed_input):
        np.random.seed(random_seed_input)
        init_embedding_vec = np.random.rand(1,self.embed_dim)
        return(init_embedding_vec)
    
    def create_embedding_matrix(self):
        num_words_to_embed = self.vocab_size+1
        embedding_matrix = np.zeros((num_words_to_embed, self.embed_dim))
    
        init_embedding_vec_for_start_token = self.create_init_vectors_for_tokens(121)
        init_embedding_vec_for_end_token = self.create_init_vectors_for_tokens(243)
        init_embedding_vec_for_unknown_token = self.create_init_vectors_for_tokens(377)
        
        embeddings_to_index_dict = self.create_word_to_embeddings_dict()
        self.check_word_embedding_dimension_correct('the')
        self.check_word_embedding_dimension_correct('a')        
        for word, i in self.word_to_index_dict.items():
            embedding_vector = embeddings_to_index_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            if embedding_vector is None and word==self.unknown_token:               
                embedding_matrix[i] = init_embedding_vec_for_unknown_token
            if embedding_vector is None and word  ==  self.sentence_start_token:
                embedding_matrix[i] = init_embedding_vec_for_start_token
            if embedding_vector is None and word==self.sentence_end_token:
                embedding_matrix[i] = init_embedding_vec_for_end_token    
        return(embedding_matrix)        
