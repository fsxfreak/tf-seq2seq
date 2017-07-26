#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import json
import random
import sys

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from data.data_iterator import BiTextIterator

import data.data_utils as data_utils
from data.data_utils import prepare_train_batch

from seq2seq_model import Seq2SeqModel

tf.app.flags.DEFINE_string('hyp_data', 'in.hyp',
  'Path to hypotheses for rescoring')
tf.app.flags.DEFINE_string('workdir', '/tmp',
  'Temporary directory to do file work')
tf.app.flags.DEFINE_string('scores', 'out.scores',
  'Path to output scores file')
tf.app.flags.DEFINE_string('source_vocabulary', 'data/europarl-v7.1.4M.de.json', 
  'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', 'data/europarl-v7.1.4M.fr.json', 
  'Path to target vocabulary')

tf.app.flags.DEFINE_string('cell_type', 'lstm', 
  'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 
  'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 1024, 
  'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 2, 
  'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 500, 
  'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 30000, 
  'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 30000, 
  'Target vocabulary size')

tf.app.flags.DEFINE_boolean('use_residual', True, 
  'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, 
  'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 
  'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 
  'Dropout probability for input/output/state units (0.0: no dropout)')


tf.app.flags.DEFINE_float('learning_rate', 0.0002, 
  'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 
  'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('max_seq_length', 50, 
  'Maximum sequence length')
tf.app.flags.DEFINE_string('optimizer', 'adam', 
  'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'model/', 
  'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 
  'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('use_fp16', False, 
  'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_boolean('average_loss', False, 
  'Normalize loss scores to length of sequence')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 
  'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 
  'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

def load_model(session, FLAGS):
  config = OrderedDict(sorted(FLAGS.__flags.items()))
  model = Seq2SeqModel(config, 'train')

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print 'Reloading model parameters..'
    model.restore(session, ckpt.model_checkpoint_path)
  else:
    raise ValueError('Checkpoint does not exist. Cannot force decode.')

  return model

def main():
  print 'Force decoding: ', FLAGS.hyp_data
  print 'Loading test data..'
  hyp = FLAGS.hyp_data
  tmp = FLAGS.workdir
  print 'Using temp directory', tmp
  hyp_src_fname = os.path.join(tmp, 
      '%s.src.%d' % (os.path.basename(hyp), int(time.time())))
  hyp_trg_fname = os.path.join(tmp, 
      '%s.trg.%d' % (os.path.basename(hyp), int(time.time())))
  print 'hyp tmp:', hyp_src_fname, hyp_trg_fname

  hyp_src = open(hyp_src_fname, 'w')
  hyp_trg = open(hyp_trg_fname, 'w')
  with open(hyp, 'r') as f:
    for line in f:
      toks = line.strip().split('\t')
      hyp_src.write('%s\n' % toks[0].strip())
      hyp_trg.write('%s\n' % toks[1].strip())
  hyp_src.close()
  hyp_trg.close()

  test_set = BiTextIterator(source=hyp_src_fname,
                            target=hyp_trg_fname,
                            source_dict=FLAGS.source_vocabulary,
                            target_dict=FLAGS.target_vocabulary,
                            batch_size=1,
                            maxlen=None,
                            n_words_source=FLAGS.num_encoder_symbols,
                            n_words_target=FLAGS.num_decoder_symbols,
                            sort_by_length=False)
  print 'Done loading test data.'

  losses = []
  with tf.Session(config=tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement, 
    log_device_placement=FLAGS.log_device_placement, 
    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

    model = load_model(sess, FLAGS)
    index = 0 
    for source_seq, target_seq in test_set:
      source, source_len, target, target_len = \
          prepare_train_batch(source_seq, target_seq)

      step_loss = model.eval(sess, 
          encoder_inputs=source, encoder_inputs_length=source_len,
          decoder_inputs=target, decoder_inputs_length=target_len)

      if index % 5000 == 0:
        print 'Rescoring hypothesis: %d' % index
      index += 1

      loss = np.sum(step_loss)
      losses.append(str(loss)) # convert to string for output to text file

  print 'Got %d scores.' % len(losses)
  with open(FLAGS.scores, 'w') as f:
    f.write('\n'.join(losses))
    f.write('\n')
  print 'Done, saved to %s.' % FLAGS.scores

if __name__ == '__main__':
  main()
