
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from params import Hyperparams
from functools import reduce


class Model(object):
    """
    The hybrid bidirectional binary encoder decoder model
    """
    def __init__(self, hyperparams, vocab_size, num_steps_encoder, num_steps_decoder, infer=False, ):
            
        assert isinstance(hyperparams, Hyperparams), 'Not an instance of Hyperparams class'

        self.num_steps_encoder = num_steps_encoder
        self.num_steps_decoder = num_steps_decoder
        self.batch_size = hyperparams.batch_size
        if infer:
            self.batch_size = 1

        self.input_placeholder_encoder = tf.placeholder(tf.int32, [None, self.num_steps_encoder])
        self.tags_placeholder_encoder = tf.placeholder(tf.float32, [None, self.num_steps_encoder, 1])
        self.input_placeholder_decoder = tf.placeholder(tf.int32, [None, self.num_steps_decoder])
        self.tags_placeholder_decoder = tf.placeholder(tf.float32, [None, self.num_steps_decoder, 1])
        self.target_placeholder = tf.placeholder(tf.int32, [None, self.num_steps_decoder])

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.use_dropout = tf.placeholder(tf.bool)
        self.num_experts = hyperparams.num_experts
        inputs_encoders = []
        inputs_decoders = []
        self.embeddings = []

        with tf.device('/cpu:0'):
            for i in range(self.num_experts):
                embedding = tf.Variable(tf.truncated_normal([vocab_size * 2, hyperparams.embedding_sizes[i]], stddev=0.05))
                inputs_encoder = tf.nn.embedding_lookup(embedding, self.input_placeholder_encoder)
                inputs_decoder = tf.nn.embedding_lookup(embedding, self.input_placeholder_decoder)

                with tf.variable_scope('tags-%d'%i):
                    inputs_encoder = tf.reshape(inputs_encoder, [-1, hyperparams.embedding_sizes[i]])
                    tags_encoder = tf.reshape(self.tags_placeholder_encoder, [-1, 1])
                    inputs_encoder = self.linear([inputs_encoder, tags_encoder], hyperparams.embedding_sizes[i])
                    inputs_encoder = tf.reshape(inputs_encoder, [-1, self.num_steps_encoder, hyperparams.embedding_sizes[i]])
                    
                    tf.get_variable_scope().reuse_variables()
                    
                    inputs_decoder = tf.reshape(inputs_decoder, [-1, hyperparams.embedding_sizes[i]])
                    tags_decoder = tf.reshape(self.tags_placeholder_decoder, [-1, 1])
                    inputs_decoder = self.linear([inputs_decoder, tags_decoder], hyperparams.embedding_sizes[i])
                    inputs_decoder = tf.reshape(inputs_decoder, [-1, self.num_steps_decoder, hyperparams.embedding_sizes[i]])

                inputs_encoders.append(inputs_encoder)
                inputs_decoders.append(inputs_decoder)
                self.embeddings.append(embedding)

        
        self.enc_states = []
        self.enc_states_rev = []
        cells = []
        with tf.variable_scope('encoder', initializer=tf.random_normal_initializer(stddev=0.1)):
            for i in range(self.num_experts):
                cell = self.make_cell(hyperparams.hidden_sizes[i], dropout_prob=hyperparams.dropout_probs[i])
                cells.append(cell)
                with tf.variable_scope(str(i+1)):
                    with tf.variable_scope('forward'):
                        enc_output, final_state_enc = tf.nn.dynamic_rnn(cell, inputs_encoders[i], dtype=tf.float32)
                    with tf.variable_scope('reverse'):
                        inputs_reversed = tf.reverse(inputs_encoders[i], tf.constant(1, shape=[1]))
                        _, final_state_enc_rev = tf.nn.dynamic_rnn(cell, inputs_reversed, dtype=tf.float32)
                    self.enc_states.append(final_state_enc)
                    self.enc_states_rev.append(final_state_enc_rev)
        
        with tf.variable_scope('decoder', initializer=tf.random_normal_initializer(stddev=0.1)):
            decoder_outputs = []
            self.expert_predictions = []
            self.decoder_outputs_raw = []
            cross_entropy_experts = []
            labels = self.target_placeholder
            for i in range(self.num_experts):
                with tf.variable_scope(str(i+1)):
                    with tf.variable_scope('forward'):
                        dec_out_f, _ = tf.nn.dynamic_rnn(cells[i], inputs_decoders[i],
                                                                        initial_state=self.enc_states[i], dtype=tf.float32)
                        dec_out_f = tf.reshape(dec_out_f, [-1, hyperparams.hidden_sizes[i]])
                    with tf.variable_scope('reverse'):
                        inputs_reversed = tf.reverse(inputs_decoders[i], tf.constant(1, shape=[1]))
                        dec_out_r, _ = tf.nn.dynamic_rnn(cells[i], inputs_reversed,
                                                                        initial_state=self.enc_states_rev[i], dtype=tf.float32)
                        dec_out_r = tf.reshape(tf.reverse(dec_out_r, tf.constant(1, shape=[1])), [-1, hyperparams.hidden_sizes[i]])
                    
                    self.decoder_outputs_raw.append(tf.reshape(dec_out_f, [-1, self.num_steps_decoder * hyperparams.hidden_sizes[i]]))
                    dec_out = tf.reshape(self.linear([dec_out_f, dec_out_r], 1),
                                                  [-1, self.num_steps_decoder])
                expert_pred = tf.sigmoid(dec_out)
                self.expert_predictions.append(expert_pred)

                ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=dec_out)
                cross_entropy_experts.append(ce)

                           

        def get_accuracy(conf_fn, thresh=0.5):
            confidences = []
            for ep in self.expert_predictions:
                #conf = tf.reduce_mean(tf.maximum(ep - 0.5, 0), axis=1) * 2
                conf = conf_fn(ep)
                confidences.append(tf.expand_dims(conf, axis=1))

            confidences_c = tf.concat(confidences, 1)
            best_indices = tf.cast(tf.argmax(confidences_c, 1), tf.int32)
            gather_indices = tf.concat([tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), 1),
                                        tf.expand_dims(best_indices, 1)], 1)
            best_predictions = tf.gather_nd(tf.stack(self.expert_predictions, axis=1), gather_indices)
            best_predictions = tf.squeeze(best_predictions)
            correct_predictions = tf.equal(tf.cast(best_predictions >= thresh, tf.int32), labels)
            correct_predictions_aligned = tf.cast(tf.reduce_all(correct_predictions, axis=1), tf.float32)
            accuracy = tf.reduce_mean(correct_predictions_aligned)
            accuracy_char = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return accuracy, confidences_c, best_predictions, correct_predictions_aligned, accuracy_char
        
        self.accuracy, self.confidences, self.predictions, _, self.accuracy_char = get_accuracy(
            lambda ep: tf.reduce_mean(tf.abs(ep - 0.5), axis=1) * 2, 0.5)
        
        self.accuracy_upper, _, _, cpa, _ = get_accuracy(
            lambda ep: tf.reduce_mean(tf.abs(ep - 0.5), axis=1) * 2, 0.4)
        
        
        cps = reduce(lambda a, x: tf.logical_or(a, tf.reduce_all(
            tf.equal(tf.cast(x >= 0.5, tf.int32), labels), axis=1)), self.expert_predictions, False)
        self.accuracy_soft =  tf.reduce_mean(tf.cast(cps, tf.float32))
        
        self.correct_values = tf.concat([tf.expand_dims(tf.cast(tf.reduce_all(tf.equal(tf.cast(x >= 0.5, tf.int32), labels),
                                                                              axis=1), tf.int32), axis=1)
                                         for x in self.expert_predictions], 1)
        self.correct_counts = reduce(lambda a, x: a + tf.cast(tf.reduce_all(
            tf.equal(tf.cast(x >= 0.5, tf.int32), labels), axis=1), tf.int32), self.expert_predictions, 0)
        

        if not infer:
            losses = []
            for ce in cross_entropy_experts:
                loss = tf.reduce_mean(ce, 1)
                losses.append(tf.expand_dims(loss, 1))
            
            losses_c = tf.concat(losses, 1)
            optimizer = tf.train.AdamOptimizer(hyperparams.learn_rate)
            self.train_op_experts = optimizer.minimize(tf.reduce_mean(losses_c, 0), global_step=self.global_step)
            self.losses = tf.reduce_sum(losses_c)
            #self.losses = tf.add_n(losses)
            #self.total_iterations = 0
    
    def make_cell(self, size, num_layers=1, dropout_prob=None):

        #cell = tf.contrib.rnn.LSTMCell(size)
        cell = tf.contrib.rnn.GRUCell(size)
        if dropout_prob:
            keep_prob = tf.cond(self.use_dropout, lambda: tf.constant(dropout_prob), lambda: tf.constant(1.0))
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)

        return cell
    
    def linear(self, inputs, output_size):
        if not nest.is_sequence(inputs):
            inputs = [inputs]

        input_size = 0
        for inp in inputs:
            shape = inp.get_shape()
            assert shape.ndims == 2, 'inputs must be 2d'
            assert shape[1].value != None, 'shape[1] cannot be None'

            input_size += shape[1].value
        #scope = tf.get_variable_scope()
        with tf.variable_scope('linear'):
            weights = tf.get_variable('weights', [input_size, output_size], 
                                      initializer=tf.random_normal_initializer(stddev=0.05))

            #print(weights.name)
            biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(0.1))
            if len(inputs) == 1:
                linear_output = tf.matmul(inputs[0], weights) + biases
            else:
                linear_output = tf.matmul(tf.concat(inputs, 1), weights) + biases

            #dropout_prob = 1.0 if infer else 0.25
            #return tf.nn.dropout(linear_output, keep_prob=dropout_prob)
            return linear_output