#!/usr/bin/python
# -*- coding: utf-8 -*-

"""This module is used for data processing

    * from the very beginning this module is supposed for rt-polarity

Todo:
    * make it can also process kaggle-movie-review dataset
    * make it can process causal classification

"""


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import os
import argparse
from hbconfig import Config


def load_into_dataset(file_names):
    """Load data from raw files to dataset

    This function uses tf.data.TextLineDataset directly

    Args:
        file_names (List): names of raw files

    Returns:
        all_labeled_data: formatted data with labels
        which format is tensorflow.python.data.ops.dataset_ops.ShuffleDataset

    Note:
        should decided BUFFER_SIZE beforehand for shuffle

    """
    
    parent_dir = os.path.join(Config.data.base_path, Config.data.raw_data_path)

    def labeler(example, index):
        return example, tf.cast(index, tf.int64)

    labeled_data_sets = []

    for i, file_name in enumerate(file_names):
        text_dir = os.path.join(parent_dir, file_name)
        
        lines_dataset = tf.data.TextLineDataset(text_dir)
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(
        Config.model.buffer_size, reshuffle_each_iteration=False)
    
    return all_labeled_data


def build_vocabulary(all_labeled_data):
    """Build word vocabulary according to labeled data

    Args:
        all_labeled_data (ShuffleDataset):
        the result got from load_into_datasets function

    Returns:
        vocabulary_set (set): word vocabulary in set

    Note:
        * tokenizer.tokenize is used to make tokens and vocabulary
        * format of all_labeled_data is tensorflow.python.data.ops.dataset_ops.ShuffleDataset

    """

    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    return vocabulary_set


def encode_labeled_data(all_labeled_data, vocabulary_set):
    """Encode text lines as numbers

    Args:
        all_labeled_data (ShuffleDataset): result got from load_into_dataset()
        vocabulary_set (set): result got from build_vocabulary()

    Returns:
        all_encoded_data (MapDataset) : encoded all text lines
        which is in form of MapDataset of tensorflow

    Note:
        tfds.features.text.TokenTextEncoder is used encode text
        tf.py_function is used to wraps a python function into a
        tensorflow op that executes it eagerly

    """

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    all_encoded_data = all_labeled_data.map(encode_map_fn)

    return all_encoded_data


def split_into_train_test_batches(all_encoded_data):
    """Split dataset into train and test batches

    Args:
        all_encoded_data (MapDataset): result from encode_labeled_data()

    Returns:
        train_data (PaddedBatchDataset): data for training
        test_data (PaddedBatchDataset): data for testing

    Note:
        TAKE_SIZE and BATCH_SIZE should be defined beforehand

    """

    train_data = all_encoded_data.skip(Config.model.take_size).shuffle(Config.model.buffer_size)
    train_data = train_data.padded_batch(Config.model.batch_size, padded_shapes=([-1], []))

    val_data = all_encoded_data.take(Config.model.take_size)
    val_data = val_data.padded_batch(Config.model.batch_size, padded_shapes=([-1], []))

    return train_data, val_data