#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from hbconfig import Config
import argparse
import atexit
import utils

from model import TextCnnRand
from data_loader_tf import load_into_dataset, build_vocabulary, encode_labeled_data, split_into_train_test_batches


class TrainModel:

    def __init__(self, model, train_data, val_data):
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=False)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Accuracy(name='val_accuracy')
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    # train the model
    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, lines, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(lines)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # test the model
    # @tf.function(experimental_relax_shapes=True)
    def val_step(self, lines, labels):
        predictions = self.model(lines)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    # generate log for tensorboard usage
    def _generate_log(self):
        # Set up summary writers to write the summaries to disk in a different logs directory
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_log_dir = 'logs/gradient_tape/' + current_time + '/validation'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        return train_summary_writer, test_summary_writer

    # Use tf.GradientTape to train the model
    def train(self):

        train_summary_writer, test_summary_writer = self._generate_log()

        # Use tf.GradientTape to train the model:
        for epoch in range(Config.train.epochs):

            for train_lines, train_labels in self.train_data:
                self.train_step(train_lines, train_labels)

            for val_lines, val_labels in self.val_data:
                self.val_step(val_lines, val_labels)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.val_accuracy.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'

            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result(),
                                  self.val_loss.result(),
                                  self.val_accuracy.result()))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()


if __name__ == '__main__':
    # get parameter from terminal
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()
    Config(args.config)

    # prepare data for model training and testing
    FILE_NAMES = ['rt-polarity-pos.txt', 'rt-polarity-neg.txt']
    all_labeled_data = load_into_dataset(FILE_NAMES)
    vocabulary_set = build_vocabulary(all_labeled_data)
    all_encoded_data = encode_labeled_data(all_labeled_data, vocabulary_set)
    train_data, val_data = split_into_train_test_batches(all_encoded_data)

    # Create an instance of the model
    vocabulary_size = len(vocabulary_set) + 1
    model = TextCnnRand(vocabulary_size)

    train_model = TrainModel(model, train_data, val_data)
    train_model.train()

    # After terminated Notification to Slack
    # atexit.register(utils.send_message_to_slack, config_name=args.config)
