"""Utility functions."""

import pickle as pkl
import json
import logging
import time

import numpy
import sys
import numpy as np
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError


if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
except (ModuleNotFoundError, ImportError) as e:
    import exception


def parse_transitions(target_dict, splitted_action=False):
    """
    converts a dictionary to edge types and labels dictionaries.
    :param target_dict: dictionary of string tokens to integers
    :param splitted_action: whether format is edge+label+"@@|" or edge+"@@|" and split\separated label + "|@@"
    :return: edge types dict, labels dict
    The returned dictionaries have numbers from 0 to number of elements in dict as values.
    Given the same edge types or labels the same dictionary will be made.
    """
    edge_end = "@@|"
    if splitted_action:
        label_start = "|@@"
        target_actions = {key: val
                          for key, val in target_dict.items() if edge_end in key}
        target_actions = reset_dict_vals(target_actions)

        target_labels = {key: val
                         for key, val in target_dict.items() if label_start in key}
        target_labels = reset_dict_vals(target_labels)
    else:
        target_actions = {key: val
                          for key, val in target_dict.items() if edge_end in key}
        actions = {key[0]: val
                   for key, val in target_actions.items()}
        actions = reset_dict_vals(actions)
        target_actions = {key: actions[key[0]]
                          for key in target_actions}

        labels = {key[1:-len(edge_end)]: i
                  for i, key in enumerate(target_actions)}
        labels = reset_dict_vals(labels)
        target_labels = {key: labels[key[1:-len(edge_end)]]
                         for key in target_actions}
    return target_actions, target_labels


def reset_dict_vals(d):
    """
    :param d: dictionary
    :return: vals are consecutive numbers starting from 0, sorted by the order of the original vals
    """
    return {key: i for i, (key, val) in enumerate(sorted(d.items(), key=lambda x: x[1]))}


def reset_dict_indexes(d):
    """
    :param d: dictionary
    :return: keys are consecutive numbers starting from 0, sorted by the order of the keys
    """
    return {i: d[key] for i, key in enumerate(sorted(d.keys()))}


def prepare_data(seqs_x, seqs_y, seq_edges_time, seq_labels_time, seq_parents_time, n_factors, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    # move edges to length major instead of batch major
    if seq_parents_time is not None:
        seq_parents_time = numpy.array(seq_parents_time)
        seq_parents_time = numpy.moveaxis(seq_parents_time, 0, -1)
    else:
        seq_parents_time = None

    if seq_edges_time is not None:
        target_edges_time = numpy.array(seq_edges_time)
        target_edges_time = numpy.moveaxis(target_edges_time, 0, -1)
        target_labels_time = numpy.array(seq_labels_time)
        target_labels_time = numpy.moveaxis(target_labels_time, 0, -1)
    else:
        target_edges_time = None
        target_labels_time = None

    # drop pairs with sentences longer than maxlen
    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        kept = []
        for i, (l_x, s_x, l_y, s_y) in enumerate(zip(lengths_x, seqs_x, lengths_y, seqs_y)):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                kept.append(i)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        if seq_parents_time is not None:
            seq_parents_time = seq_parents_time[..., kept]
        if seq_edges_time is not None:
            target_edges_time = target_edges_time[:, :, :, kept]
            target_labels_time = target_labels_time[:, :, :, kept]
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1
    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = list(zip(*s_x))
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.

    return x, x_mask, y, y_mask, target_edges_time, target_labels_time, seq_parents_time
    # return x, x_mask, y, y_mask, target_edges, target_labels, target_edges_time, target_labels_time


def _parent_row_to_attention(row):
    # np.zeros_like(row) + -1e9 * row > np.arange(len(row))
    logging.info(f"row? {row} {row.shape}")
    return np.array([0 if x <= i else -1e9 for i, x in enumerate(row)])


def times_to_parents(times, repeat=1):
    """
    Converts from times [] to parents input for the network
    :param times:
    :param repeat:
    :return: attention changes with shape [from_tok, to_tok, sent]
    """
    # start_time = time.time()
    # logging.info(f"times.shape {times[0].shape} {times[0]} repeat {repeat}")
    if len(times.shape) == 1:
        max_sen_len = repeat
        tmp = []
        for sent_times in times:
            pad = max_sen_len - len(sent_times)
            tmp.append(np.pad(sent_times, ((0, pad), (0, pad)), mode="constant", constant_values=float("inf")))
        times = tmp
        times = np.array(times)
        times = np.transpose(times, [1, 2, 0])  # time major


    # new_start_time = time.time()
    attention = np.empty(times.size * repeat)  # mask per repetition
    flat_times = times.flatten()
    for repetition in range(repeat):
        attention[repetition::repeat] = np.where(flat_times <= repetition, 0, float("inf"))
    # logging.info(f"new_times.shape {len(attention[0])}, {attention[0].shape} repeat {repeat}")

    # interleave (make same sentences on different words one after the other)
    attention_shape = times.shape
    shape = (attention_shape[0], attention_shape[1], attention_shape[2] * repeat)
    # attention = np.stack(attention, axis=0).reshape(shape)
    attention = attention.reshape(shape)
    # new_mid_time = time.time()
    # logging.info(f"in new loop {new_mid_time - new_start_time}")


    # new_start_time = time.time()
    # attention = [] # mask per repetition
    # for repetition in range(repeat):
    #     new_time = np.where(times <= repetition, 0, float("inf"))
    #     attention.append(new_time)
    # # logging.info(f"new_times.shape {len(attention[0])}, {attention[0].shape} repeat {repeat}")
    #
    # # interleave (make same sentences on different words one after the other)
    # attention_shape = attention[0].shape
    # shape = (attention_shape[0], attention_shape[1], attention_shape[2] * repeat)
    # attention = np.stack(attention, axis=0).reshape(shape)
    # attention = attention.reshape(shape)
    # # new_mid_time = time.time()
    # # logging.info(f"in new loop {new_mid_time - new_start_time}")

    # logging.info(f"attention.shape {attention.shape} {attention} repeat {repeat}")
    # times = np.repeat(times, repeat, axis=-1)
    # shape = times.shape
    # mid_time = time.time()
    # # logging.info(f"in first loop {mid_time - start_time}")
    # for from_tok, to_tok, sent in np.ndindex(shape):
    #     if times[from_tok, to_tok, sent] <= sent % repeat:
    #         times[from_tok, to_tok, sent] = 0
    #     else:
    #         times[from_tok, to_tok, sent] = float("inf")
    # end_time = time.time()
    # logging.info(f"times.shape {times.shape} {times} repeat {repeat}")
    # # logging.info(f"in second loop {end_time - mid_time}")
    # assert np.all(times == attention)
    return attention


def times_to_input(times, timesteps):
    idx = []
    for sentence_num in range(times.shape[-1]):
        sentence = times[:, :, :, sentence_num]
        sparse = array_to_sparse_tensor(sentence, feeding=True)
        cur_idx = np.tile(sparse.indices, [timesteps, 1])
        cur_vals = np.tile(sparse.values, [timesteps])
        num_indices = sparse.indices.shape[0]  # = cur_idx.shape[0] / timesteps
        cur_timesteps = np.array([i // num_indices for i in range(cur_idx.shape[0])])
        cur_idx = np.c_[cur_idx, sentence_num * timesteps + cur_timesteps]  # add batch column
        cond = cur_vals < (cur_timesteps + 1)
        cur_idx = cur_idx[cond, :]
        idx.append(cur_idx)

    idx = np.concatenate(idx, axis=0)
    values = np.ones((idx.shape[0],), dtype=np.float32)
    shape = times.shape * np.array((1, 1, 1, timesteps))
    sparse = tf.compat.v1.SparseTensorValue(indices=idx, values=values, dense_shape=shape)
    return sparse


def array_to_sparse_tensor(ar, base_val=float("inf"), feeding=False):
    if not feeding:
        ar = tf.convert_to_tensor(ar)
    sparse = dense_to_sparse_tensor(ar, base_val=base_val, feeding=feeding)

    return sparse


def dense_to_sparse_tensor(dns, base_val=float("inf"), feeding=False):
    """
    convert a dense tensor to a sparse one
    :param dns: a tensor (currently dynamic shape is not supported)
    :param base_val: which values to remove from the tensor (default float("inf")
    :return:
    """

    if feeding:
        sparse_init = tf.compat.v1.SparseTensorValue
        idx = np.array(np.where(dns != base_val)).transpose()
        shape = dns.shape
        values = dns[dns != base_val].flatten()
    else:
        idx = tf.where(tf.not_equal(dns, base_val))
        values = tf.gather_nd(dns, idx)
        shape = tf.cast(tf.shape(dns), tf.int64)
        sparse_init = tf.compat.v1.SparseTensor
    sparse = sparse_init(indices=idx, values=values, dense_shape=shape)
    return sparse


def load_dict(filename, model_type):
    print(f"loading dict from {filename}")
    try:
        # build_dictionary.py writes JSON files as UTF-8 so assume that here.
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        # FIXME Should we be assuming UTF-8?
        with open(filename, 'r', encoding='utf-8') as f:
            d = pkl.load(f)

    # The transformer model requires vocab dictionaries to use the new style
    # special symbols. If the dictionary looks like an old one then tell the
    # user to update it.
    if model_type == 'transformer' and ("<GO>" not in d or d["<GO>"] != 1):
        logging.error('you must update \'{}\' for use with the '
                      '\'transformer\' model type. Please re-run '
                      'build_dictionary.py to generate a new vocabulary '
                      'dictionary.'.format(filename))
        sys.exit(1)

    return d


def seq2words(seq, inverse_dictionary, join=True):
    seq = numpy.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_dictionary],
                             join)


def factoredseq2words(seq, inverse_dictionaries, join=True):
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    eos_reached = False
    for i, w in enumerate(seq):
        if eos_reached:
            break
        factors = []
        for j, f in enumerate(w):
            if f == 0:
                eos_reached = True
                break
            elif f in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][f])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words


def reverse_dict(dictt):
    keys, values = list(zip(*list(dictt.items())))
    r_dictt = dict(list(zip(values, keys)))
    return r_dictt


def load_dictionaries(config):
    model_type = config.model_type
    source_to_num = [load_dict(d, model_type) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict, model_type)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target


def read_all_lines(config, sentences, batch_size):
    source_to_num, _, _, _ = load_dictionaries(config)

    if config.source_vocab_sizes != None:
        assert len(config.source_vocab_sizes) == len(source_to_num)
        for d, vocab_size in zip(source_to_num, config.source_vocab_sizes):
            if vocab_size != None and vocab_size > 0:
                for key, idx in list(d.items()):
                    if idx >= vocab_size:
                        del d[key]

    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            if config.factors == 1:
                w = [source_to_num[0][w] if w in source_to_num[0] else 2]
            else:
                w = [source_to_num[i][f] if f in source_to_num[i] else 2
                     for (i, f) in enumerate(w.split('|'))]
                if len(w) != config.factors:
                    raise exception.Error(
                        'Expected {0} factors, but input words has {1}\n'.format(
                            config.factors, len(w)))
            line.append(w)
        lines.append(line)
    lines = numpy.array(lines)
    lengths = numpy.array([len(l) for l in lines])
    idxs = lengths.argsort()
    lines = lines[idxs]

    # merge into batches
    batches = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        batches.append(batch)

    return batches, idxs

if __name__ == '__main__':

    a = np.arange(24,dtype=np.float32).reshape((2,3,4))
    times_to_parents(a, repeat=4)