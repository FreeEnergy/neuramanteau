
from __future__ import print_function

import tensorflow as tf
import numpy as np

from data_provider import DataProvider
from model_ensemble_binary_bidir import Model
from params import Hyperparams
from vocab_provider import VocabProvider
import os
import sys
import argparse

TRAIN = 'train'
VALIDAION = 'validation'
TEST = 'test'

default_params = Hyperparams(
    batch_size=100,
    num_epochs=12,
    learn_rate=0.006,
    num_experts=6,
    embedding_sizes=[128]*6,
    hidden_sizes=[128]*6,
    dropout_probs=[0.75]*6
)

default_partitions = [
    (TRAIN, 0.75),
    (VALIDAION, 0.15),
    (TEST, 0.1)
]

def setup_model_path(model_name):
    model_dir = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model')
    return model_path, model_dir

def optimize(session, model, provider, epoch_size, batch_size):
    total_loss = 0
    for i, (x, y, z, u, v) in enumerate(provider.iterate(epoch_size, batch_size, TRAIN)):
        loss_value, _ = session.run([model.losses, model.train_op_experts], 
                    feed_dict={model.input_placeholder_encoder:x,
                                                model.input_placeholder_decoder:y,
                                                model.target_placeholder:z,
                                                model.tags_placeholder_encoder:u,
                                                model.tags_placeholder_decoder:v,
                                                model.use_dropout: True
                                                })
        total_loss += loss_value
        if i % (epoch_size // 10) == 9:
            print("%.3f" % (total_loss/(i + 1)), end=' ')
            sys.stdout.flush()

    print()
    return total_loss/epoch_size

def validation_loss(session, model, provider, batch_size):
    total_loss = 0
    iterations = 0
    for x,y,z,u,v in provider.iterate_seq(batch_size, VALIDAION):
        loss_value = session.run(model.losses, feed_dict={
                                        model.input_placeholder_encoder:x,
                                        model.input_placeholder_decoder:y,
                                        model.tags_placeholder_encoder:u,
                                        model.tags_placeholder_decoder:v,
                                        model.target_placeholder:z,
                                        model.use_dropout: False})
        total_loss += loss_value
        iterations += 1

    return total_loss/iterations

def get_accuracy(session, model, provider, batch_size, partition_name):
    correct = 0
    correct_upper = 0
    correct_char = 0
    correct_soft = 0
    count = 0

    for x,y,z,u,v in provider.iterate_seq(model.batch_size, partition_name):
        accuracy, accuracy_upper, accuracy_char, accuracy_soft  = session.run([model.accuracy, model.accuracy_upper, model.accuracy_char, 
                                                                                model.accuracy_soft],
                                                                feed_dict={
                                                                model.input_placeholder_encoder:x,
                                                                model.input_placeholder_decoder:y,
                                                                model.tags_placeholder_encoder:u,
                                                                model.tags_placeholder_decoder:v,
                                                                model.target_placeholder:z,
                                                                model.use_dropout: False})

        correct += accuracy
        correct_upper += accuracy_upper
        correct_char += accuracy_char
        correct_soft += accuracy_soft
        count += 1

    return correct/count, correct_upper/count, correct_char/count, correct_soft/count


def train(provider, hyperparams, save_model_name, resume_checkpoint, checkpoint=None):
    model_path, _ = setup_model_path(save_model_name)
    maxlen_source, _ = provider.seq_sizes()
    train_size = provider.dataset_size(TRAIN)
    epoch_size = train_size // hyperparams.batch_size
    if epoch_size < 100:
        epoch_size = 100
    print('Epoch size: ', epoch_size)

    with tf.Graph().as_default():
        print('Loading model: ', save_model_name)
        train_model = Model(hyperparams, vocab_size=provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source)
        saver = tf.train.Saver(max_to_keep=12)
        with tf.Session() as session:
            if resume_checkpoint:
                if checkpoint:
                    saver.restore(session, model_path + '-' + str(checkpoint))
                else:
                    saver.restore(session, tf.train.latest_checkpoint(model_dir))
            else:
                session.run(tf.global_variables_initializer())
            
            print('Start training')
            total_iterations = train_model.global_step.eval(session=session)

            for i in range(hyperparams.num_epochs):
                print('Epoch %d' % (total_iterations//epoch_size + 1))

                loss = optimize(session, train_model, provider, epoch_size, hyperparams.batch_size)
                valid_loss = validation_loss(session, train_model, provider, hyperparams.batch_size)
                valid_acc, valid_acc_o, valid_acc_c, valid_acc_s = get_accuracy(session, train_model, provider, hyperparams.batch_size, VALIDAION)

                total_iterations = train_model.global_step.eval(session=session)

                print('After %d iterations, loss = %.3f' % (total_iterations, loss))
                print('Validation loss = %.3f' % (valid_loss))
                print('Validation accuracy = %.3f' % (valid_acc))
                print('Validation accuracy(offset) = %.3f' % (valid_acc_o))
                print('Validation accuracy(any expert) = %.3f' % (valid_acc_s))
                saver.save(session, model_path, global_step=train_model.global_step)
            
            print('Training completed')


def evaluate_accuracy_conf(provider, hyperparams, load_model_name, partition_name=None, checkpoint=None):

    model_path, model_dir = setup_model_path(load_model_name)
    with tf.Graph().as_default():
        maxlen_source, _ = provider.seq_sizes()
        print('Creating model..')
        model = Model(hyperparams, vocab_size=provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source)
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('Loading data..')
            if checkpoint:
                saver.restore(session, model_path + '-' + str(checkpoint))
            else:
                saver.restore(session, tf.train.latest_checkpoint(model_dir))
            print('Evaluating... (Confidence measure based accuracy)')
            acc, acc_o, acc_c, acc_s = get_accuracy(session, model, provider, hyperparams.batch_size, partition_name)
            print('Accuracy = %.3f' % (acc))
            print('Accuracy(offset) = %.3f' % (acc_o))
            print('Accuracy(char) = %.3f' % (acc_c))
            print('Accuracy(any expert) = %.3f' % (acc_s))

def evaluate_accuracy_voted(provider, hyperparams, load_model_name, partition_name=None, checkpoint=None, show_agreements=False):
    model_path, model_dir = setup_model_path(load_model_name)
    with tf.Graph().as_default():
        maxlen_source, _ = provider.seq_sizes()
        hyperparams.batch_size = 1
        print('Creating model..')
        model = Model(hyperparams, vocab_size=provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source)
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('Loading data..')
            if checkpoint:
                saver.restore(session, model_path + '-' + str(checkpoint))
            else:
                saver.restore(session, tf.train.latest_checkpoint(model_dir))


            src, tgt = provider.get_texts(partition_name)
                
            count = 0
            tot_acc_1 = tot_acc_2 = tot_acc_3 = 0
            tot_agree = 0
            data_size = provider.dataset_size(partition_name)
            print('Evaluating... (Weighted voting based accuracy)')
            for i, (x,y,z,u,v) in enumerate(provider.iterate_seq(hyperparams.batch_size, partition_name)):
                pred_value_experts, conf = session.run([model.expert_predictions, model.confidences], feed_dict={
                                                model.input_placeholder_encoder:x,
                                                model.input_placeholder_decoder:y,
                                                model.tags_placeholder_encoder:u,
                                                model.tags_placeholder_decoder:v,
                                                model.use_dropout: False})
                predictions = {}
                conf = np.squeeze(conf)
                w = src[i]
                t = tgt[i]
                count += 1
                if not show_agreements and i % (data_size // 10) == 9:
                    print('%d%%' % ((i/data_size) * 100), end=' ')
                    sys.stdout.flush()

                    
                for i, preds in enumerate(pred_value_experts):
                    s = ''.join([w[k] if b >= 0.5 and k < len(w) else '' for k, b in enumerate(np.squeeze(preds))])
                    if s in predictions:
                        predictions[s] += conf[i]
                    else:
                        predictions[s] = conf[i]
                        
                top_preds = sorted(predictions, key=predictions.get, reverse=True)
                
                if t == top_preds[0]:
                    tot_acc_1 += 1
                if t in top_preds[:2]:
                    tot_acc_2 += 1
                if t in top_preds[:3]:
                    tot_acc_3 += 1
                    
                tot_agree += (1 - len(predictions)/hyperparams.num_experts)
                
                if show_agreements:
                    agree = t in top_preds
                    print(t, ':', 'agree' if agree else 'disagree')
            
            print()
            print('%s voted accuracy-1 = %.3f' % (partition_name, tot_acc_1/count))
            print('%s voted accuracy-2 = %.3f' % (partition_name, tot_acc_2/count))
            print('%s voted accuracy-3 = %.3f' % (partition_name, tot_acc_3/count))
            print('%s agreement = %.3f' % (partition_name, tot_agree/count))   


def expert_stats(provider, hyperparams, load_model_name, partition_name=None, checkpoint=None):
    model_path, model_dir = setup_model_path(load_model_name)
    with tf.Graph().as_default():
        maxlen_source, _ = provider.seq_sizes()
        print('Creating model..')
        model = Model(hyperparams, vocab_size=provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source)
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('Loading data..')
            if checkpoint:
                saver.restore(session, model_path + '-' + str(checkpoint))
            else:
                saver.restore(session, tf.train.latest_checkpoint(model_dir))

            
            cumul = []
            val_array = []
            count = 0
            for x,y,z,u,v in provider.iterate_seq(hyperparams.batch_size, partition_name):
                counts_val, values  = session.run([model.correct_counts, model.correct_values], feed_dict={
                                                model.input_placeholder_encoder:x,
                                                model.input_placeholder_decoder:y,
                                                model.tags_placeholder_encoder:u,
                                                model.tags_placeholder_decoder:v,
                                                model.target_placeholder:z,
                                                model.use_dropout: False})

                val_array.append(np.squeeze(values))
                cumul.append(np.squeeze(counts_val))
                count += hyperparams.batch_size

            counts = [0] * (hyperparams.num_experts + 1)
            for c in np.nditer(np.concatenate(cumul)):
                counts[c] += 1
                
            val = np.concatenate(val_array)
            print('Correct Predictions Distribution (0-%d):'%hyperparams.num_experts, counts)
            print('Expert Accuracy:', np.sum(val, 0)/count)

def show_errors(provider, hyperparams, load_model_name, partition_name=None, checkpoint=None, check_experts=False, show_all=False):
    model_path, model_dir = setup_model_path(load_model_name)
    with tf.Graph().as_default():
        maxlen_source, _ = provider.seq_sizes()
        hyperparams.batch_size = 1
        print('Creating model..')
        model = Model(hyperparams, vocab_size=provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source)
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('Loading data..')
            if checkpoint:
                saver.restore(session, model_path + '-' + str(checkpoint))
            else:
                saver.restore(session, tf.train.latest_checkpoint(model_dir))
            
            count = 0
            errors = 0
            src, tgt = provider.get_texts(partition_name)

            for i, (x,y,z,u,v) in enumerate(provider.iterate_seq(hyperparams.batch_size, partition_name)):
                pred_values, pred_values_experts, conf = session.run([model.predictions, model.expert_predictions, model.confidences], feed_dict={
                                                model.input_placeholder_encoder:x,
                                                model.input_placeholder_decoder:y,
                                                model.tags_placeholder_encoder:u,
                                                model.tags_placeholder_decoder:v,
                                                model.use_dropout: False})
                match = False
                predictions = {}
                top_preds = []

                w = src[i]
                t = tgt[i]
                count += 1
                conf = np.squeeze(conf)
                
                if check_experts:
                    for pred in pred_values_experts:
                        s = ''.join([w[k] if b >= 0.5 and k < len(w) else '' for k, b in enumerate(np.squeeze(pred))])
                        top_preds.append(s)
                        if s == t:
                            match = True
                            break
                else:
                    for i, preds in enumerate(pred_values_experts):
                        s = ''.join([w[k] if b >= 0.5 and k < len(w) else '' for k, b in enumerate(np.squeeze(preds))])
                        if s in predictions:
                            predictions[s] += conf[i]
                        else:
                            predictions[s] = conf[i]
                        
                    top_preds = sorted(predictions, key=predictions.get, reverse=True)[:2]
                    match = True if t in top_preds else False

                
                if not match:
                    errors += 1
                
                if not show_all:
                    if not match:
                        print(w,'| t =',t,'| p =', set(top_preds))
                else:
                    print(w,'| t =',t,'| p =', set(top_preds), match)
            
            print('Total errors: %d/%d' % (errors, count))

def sample(hyperparams, input_strings, data_provider, load_model_name, checkpoint=None, top_k=3, dominances=None):
    model_path, model_dir = setup_model_path(load_model_name)
    w1, w2 = input_strings

    maxlen_source, _ = data_provider.seq_sizes()

    if dominances and len(dominances) == 2:
        x, y, u, v = data_provider.map_inputs(w1, w2, d1=dominances[0], d2=dominances[1])
    else:
        x, y, u, v = data_provider.map_inputs(w1, w2)

    with tf.Graph().as_default():
        print('Loading model: ', load_model_name)
        hyperparams.batch_size = 1
        model = Model(hyperparams, data_provider.vocab_size(), num_steps_encoder=maxlen_source, num_steps_decoder=maxlen_source, infer=True)
        saver = tf.train.Saver()
        with tf.Session() as session:
            print('Loading data..')
            if checkpoint:
                saver.restore(session, model_path + '-' + str(checkpoint))
            else:
                saver.restore(session, tf.train.latest_checkpoint(model_dir))
            
            def print_result(pred_value):
                bs = np.split(np.squeeze(pred_value), num_steps_decoder)
                w = w1 + ' ' + w2
                s = ''.join([w[i] if b >= 0.5 and i < len(w) else '' for i, b in enumerate(bs)])
                print(s)


            pred_value_experts, conf = session.run([model.expert_predictions, model.confidences], feed_dict={
                                                model.input_placeholder_encoder:x,
                                                model.input_placeholder_decoder:y,
                                                model.tags_placeholder_encoder:u,
                                                model.tags_placeholder_decoder:v,
                                                model.use_dropout: False})
            predictions = {}
            conf = np.squeeze(conf)
            w = w1 + ' ' + w2

                
            for i, preds in enumerate(pred_value_experts):
                s = ''.join([w[k] if b >= 0.5 and k < len(w) else '' for k, b in enumerate(np.squeeze(preds))])
                if s in predictions:
                    predictions[s] += conf[i]
                else:
                    predictions[s] = conf[i]
                    
            top_preds = sorted(predictions, key=predictions.get, reverse=True)
            print(top_preds[:top_k])
            



def get_options():

    parser = argparse.ArgumentParser(description='RNN based lexical blender.')
    
    parser.add_argument('task', help='which task to run', choices=['train', 'eval', 'sample', 'stats', 'errors'])
    parser.add_argument('--data', help='data file for train', dest='data_path')
    parser.add_argument('--vocab', help='vocabulary file', dest='vocab_path', default='chars.vocab')
    parser.add_argument('--no-partitions', help='do not partition the data', action='store_true', dest='no_partitions')
    parser.add_argument('--partitions', help='partition fractions for train/validation/test', nargs=3, dest='partitions', type=float, 
                        metavar=('TRAIN_FRAC', 'VALIDATION_FRAC', 'TEST_FRAC'))
    parser.add_argument('--name', help='name of the model', dest='model_name', default='ensemble_binary_bidir')
    parser.add_argument('--checkpoint', help='checkpoint number(iterations) to load', dest='checkpoint')
    parser.add_argument('--add-dominance', help='add dominance markers', dest='add_dominance', action='store_true')
    
    #Hyperparams
    parser.add_argument('--batch-size', help='batch size (%(default)d)', dest='batch_size', type=int, default=default_params.batch_size)
    parser.add_argument('--num-epochs', help='number of epochs (%(default)d)', dest='num_epochs', type=int, default=default_params.num_epochs)
    parser.add_argument('--learn-rate', help='learning rate of optimizer (%(default)f)', dest='learn_rate', type=float, default=default_params.learn_rate)
    parser.add_argument('--num-experts', help='number of experts in ensemble (%(default)d)', dest='num_experts', type=int, default=default_params.num_experts)
    parser.add_argument('--embedding-size', help='character embedding size (%(default)d)', dest='embedding_size', type=int, default=default_params.embedding_sizes[0])
    parser.add_argument('--hidden-size', help='GRU hidden size (%(default)d)', dest='hidden_size', type=int, default=default_params.hidden_sizes[0])
    parser.add_argument('--dropout-prob', help='dropout keep probability (%(default)f)', dest='dropout_prob', type=float, default=default_params.dropout_probs[0])
    parser.add_argument('--resume', help='Resume from last checkpoint', dest='resume_checkpoint', action='store_true')

    #For eval task
    parser.add_argument('--accuracy-type', help='type of accuracy measure (%(default)s)', choices=['conf', 'voted'], dest='accuracy_type', default='voted')
    parser.add_argument('--partition-name', help='name of the partition', choices=['train', 'validation', 'test'], dest='partition_name')

    #For sample task
    parser.add_argument('--source-words', help='source words to blend', nargs=2, dest='source_words', metavar=('FIRST_WORD', 'SECOND_WORD'))
    parser.add_argument('--num-predictions', help='number of top predictions (%(default)d)', dest='num_predictions', type=int, default=3)
    parser.add_argument('--dominances', help='set dominance for each word', choices=[0, 1], type=int, nargs=2, dest='dominances', metavar=('D1', 'D2'))

    #For stats task
    # Nothing

    #For errors task
    parser.add_argument('--check-experts', help='check all experts to determine errors', action='store_true', dest='check_experts')
    parser.add_argument('--show-all', help='show all examples', action='store_true', dest='show_all')


    return parser.parse_args()

def get_params(options):
    return Hyperparams(
        batch_size=options.batch_size,
        num_epochs=options.num_epochs,
        learn_rate=options.learn_rate,
        num_experts=options.num_experts,
        embedding_sizes=[options.embedding_size] * options.num_experts,
        hidden_sizes=[options.embedding_size] * options.num_experts,
        dropout_probs=[options.dropout_prob] * options.num_experts
    )

def get_partitions(options):
    if not options.no_partitions:
        partitions = default_partitions
        if options.partitions is not None:
            assert len(options.partitions) == 3, 'You must provide train/validation/test partition fractions'
            for i, frac in enumerate(options.partitions):
                partitions[i][1] = frac
    else:
        partitions = None

    return partitions

def main():
    options = get_options()

    if options.task == 'train':
        assert options.data_path is not None, 'You must provide data path for training'
        
        params = get_params(options)
        partitions = get_partitions(options)

        print(params)
        print('Please use above params for eval/sample/stats/errors on same model')

        vocab_provider = VocabProvider(options.vocab_path, options.data_path)
        data_provider = DataProvider(options.data_path, vocab_provider, partitions=partitions, add_dominance=options.add_dominance)
        train(data_provider, params, options.model_name, options.resume_checkpoint, options.checkpoint)

    elif options.task == 'eval':
        assert options.data_path is not None, 'You must provide data path for evaluation'
        
        params = get_params(options)
        partitions = get_partitions(options)

        vocab_provider = VocabProvider(options.vocab_path)
        data_provider = DataProvider(options.data_path, vocab_provider, partitions=partitions, add_dominance=options.add_dominance)

        if options.accuracy_type == 'conf':
            evaluate_accuracy_conf(data_provider, params, options.model_name, options.partition_name, options.checkpoint)
        elif options.accuracy_type == 'voted':
            evaluate_accuracy_voted(data_provider, params, options.model_name, options.partition_name, options.checkpoint)

    elif options.task == 'sample':
        assert options.data_path is not None, 'You must provide data path for sampling'
        assert options.source_words is not None, 'You must provide source-words for sampling'
        assert len(options.source_words) == 2, 'You must provide exactly 2 source words'

        params = get_params(options)
        vocab_provider = VocabProvider(options.vocab_path)
        partitions = get_partitions(options)
        data_provider = DataProvider(options.data_path, vocab_provider, partitions=partitions, add_dominance=options.dominances is not None)

        sample(params, options.source_words, data_provider, options.model_name, options.checkpoint, top_k=options.num_predictions, dominances=options.dominances)

    elif options.task == 'stats':
        assert options.data_path is not None, 'You must provide data path for evaluation'

        params = get_params(options)
        partitions = get_partitions(options)

        vocab_provider = VocabProvider(options.vocab_path)
        data_provider = DataProvider(options.data_path, vocab_provider, partitions=partitions, add_dominance=options.add_dominance)

        expert_stats(data_provider, params, options.model_name, options.partition_name, options.checkpoint)

    elif options.task == 'errors':
        assert options.data_path is not None, 'You must provide data path for evaluation'

        params = get_params(options)
        partitions = get_partitions(options)

        vocab_provider = VocabProvider(options.vocab_path)
        data_provider = DataProvider(options.data_path, vocab_provider, partitions=partitions, add_dominance=options.add_dominance)

        show_errors(data_provider, params, options.model_name, options.partition_name, options.checkpoint, options.check_experts, show_all=options.show_all)






if __name__ == "__main__":
    main()