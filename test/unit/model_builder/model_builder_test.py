#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import nose
import os
import time
import numpy as np
from text_processor.text_processor import KerasTextListPreparer
import keras.callbacks
import keras_model.model_builder as ModelBuilder
from keras_model.keras_model import ConvolutionModel,WordEmbedder
from keras_model.model_builder import ModelBuilder as ModelBuilder


class TestModelBuilder(unittest.TestCase):
    
    def setUp(self):
        self.conf = {'corpus_filename':'testing_sentences.pickle',
                     'labels_filename':'testing_labels.pickle',
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
                            'vocab_size':50,
                            'num_timesteps':8,
                            'embed_dim':100,
                            'model_type':'CNN-rand'
                            }}

        self.model_builder_obj = ModelBuilder(
                                            self.conf,
                                            KerasTextListPreparer,
                                            ConvolutionModel,
                                            WordEmbedder)
        
    def test_model_builder_object_returns_constructor_params(self):        
        self.assertEqual(self.model_builder_obj.corpus_filename, 'testing_sentences.pickle')
        self.assertEqual(self.model_builder_obj.labels_filename, 'testing_labels.pickle')
        self.assertEqual(
                        self.model_builder_obj.text_proc_params,
                         {'sentence_start_token':'SENTENCE_START',
                           'sentence_end_token':'SENTENCE_END',
                           'unknown_token':'UNKNOWN_TOKEN',
                           'pad_token':'PAD_TOKEN'
                           })              
        self.assertEqual(
                        self.model_builder_obj.model_params,
                         {'vocab_size':50,
                           'num_timesteps':8,
                           'embed_dim':100,
                           'model_type':'CNN-rand'
                           })
        self.assertEqual(self.model_builder_obj.word_embedding_params,
                         {'embedding_directory':'/Volumes/GLOVE/glove.6B',
                          'embedding_file':'glove.6B.100d.txt'
                          })                       
       
    def test_load_corpus(self):
        text_list_test = self.model_builder_obj.load_corpus()
        print(text_list_test[0:2])
        self.assertEqual(text_list_test[0:2],
                         [
                          'A very very very slow-moving aimless movie about a distressed drifting young man.', 
                          'Not sure who was more lost - the flat characters or the audience nearly half of whom walked out.'
                          ])
    
    def test_load_labels(self):
        labels_test = self.model_builder_obj.load_labels()
        print(labels_test[0:2])
        self.assertTrue(np.array_equal(labels_test[0:2],np.array([0,0])))

    def test_train_model(self):
        self.assertIsInstance(self.model_builder_obj.trained_conv_model,keras.callbacks.History)
    
    def test_adding_embed_matrix_updates_model_params(self):
        embed_conf = {'corpus_filename':'testing_sentences.pickle',
                     'labels_filename':'testing_labels.pickle',
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
                            'vocab_size':50,
                            'num_timesteps':8,
                            'embed_dim':100,
                            'model_type':'CNN-non-static'
                            }}

        model_builder_with_pretrained_emb_obj = ModelBuilder(
                                                    embed_conf,
                                                    KerasTextListPreparer,
                                                    ConvolutionModel,
                                                    WordEmbedder)
        self.assertTrue(
                'embedding_matrix' in model_builder_with_pretrained_emb_obj.model_params.keys())
        
    def test_create_filename(self):
        filename_test = self.model_builder_obj.create_filename('model_results.pickle')
        self.assertEqual(filename_test,time.strftime("%d_%m_%Y") + '_' + 'model_results.pickle')

    @staticmethod
    def remove_file_from_dir(filename_input):
        if os.path.exists(filename_input):
            os.remove(filename_input)
            
    def test_save_training_data(self):
        filename_x = time.strftime("%d_%m_%Y") + '_' + 'x_train_all_samples.npy'
        filename_y = time.strftime("%d_%m_%Y") + '_' + 'y_train_all_samples.npy'
        self.remove_file_from_dir(filename_x)
        self.remove_file_from_dir(filename_y)
        self.model_builder_obj.save_training_data()
        self.assertTrue(os.path.exists(filename_x))
        self.assertTrue(os.path.exists(filename_y))

    def test_save_word_to_index_dict(self):
        filename_dict = time.strftime("%d_%m_%Y") + '_' + 'word_to_index_dict.pkl'
        self.remove_file_from_dir(filename_dict)
        self.model_builder_obj.save_word_to_index_dict()
        self.assertTrue(os.path.exists(filename_dict))
        
    def test_save_model(self):
        filename_path = time.strftime("%d_%m_%Y") + '_'+ 'conv_model.h5'
        self.remove_file_from_dir(filename_path)
        self.model_builder_obj.save_model()
        self.assertTrue(os.path.exists(filename_path))

    def test_save_model_results(self):
        current_date = time.strftime("%d_%m_%Y")
        filename_path_results = current_date+'_model_results.pickle'
        self.remove_file_from_dir(filename_path_results)
        self.model_builder_obj.save_model_results()
        self.assertTrue(os.path.exists(filename_path_results))
       
        
if __name__=='__main__':
    nose.run(defaultTest=__name__)
