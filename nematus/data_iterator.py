import sys
import numpy
import logging

import gzip

from django.contrib.sitemaps.views import x_robots_tag
from parsing.corpus import extract_text_from_combined_tokens
from parsing.corpus import ConllSent
from util import reset_dict_vals
import numpy as np
import subprocess

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .util import load_dict, parse_transitions
    from . import shuffle
except (ModuleNotFoundError, ImportError) as e:
    from util import load_dict, parse_transitions
    import shuffle


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding="UTF-8")
    return open(filename, mode, encoding="UTF-8")


class FileWrapper(object):

    def __init__(self, fname, lines_file_path):
        self.pos = 0
        self.lines = fopen(fname).readlines()
        self.lines = numpy.array(self.lines, dtype=numpy.object)
        if lines_file_path is not None:
            self.line_nums = fopen(lines_file_path).readlines()
            self.line_nums = numpy.array(self.line_nums)
            self.lines = self.lines[self.line_nums]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.lines):
            raise StopIteration
        l = self.lines[self.pos]
        self.pos += 1
        return l

    def reset(self):
        self.pos = 0

    def seek(self, pos):
        assert pos == 0
        self.pos = 0

    def readline(self):
        return next(self)

    def shuffle_lines(self, perm):
        self.lines = self.lines[perm]
        self.pos = 0

    def __len__(self):
        return len(self.lines)


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, target,
                 source_dicts, target_dict,
                 model_type,
                 batch_size=128,
                 maxlen=100,
                 source_vocab_sizes=None,
                 target_vocab_size=None,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 token_batch_size=0,
                 keep_data_in_memory=False,
                 remove_parse=False,
                 preprocess_script=None,
                 target_graph=False,
                 target_semantic_graph=False,
                 target_labels_num=None,
                 splitted_action=False,
                 ignore_empty=False,
                 lines_file_path=None):
        self.preprocess_script = preprocess_script
        self.source_orig = source
        self.target_orig = target
        self.ignore_empty = ignore_empty
        self.lines_file_path = lines_file_path
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
        if keep_data_in_memory:
            self.source, self.target = FileWrapper(source, lines_file_path), FileWrapper(target, lines_file_path)
            if shuffle_each_epoch:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
        elif shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.jointly_shuffle_files(
                [self.source_orig, self.target_orig], self.lines_file_path, temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict, model_type))
        self.target_dict = load_dict(target_dict, model_type)
        self.target_graph = target_graph
        self.target_labels_num = target_labels_num
        if self.target_graph:
            self.splitted_action = splitted_action
            self.target_actions, self.target_labels = parse_transitions(self.target_dict, self.splitted_action)
        else:
            self.target_labels = None
            self.target_actions = None

        # Determine the UNK value for each dictionary (the value depends on
        # which version of build_dictionary.py was used).

        def determine_unk_val(d):
            if '<UNK>' in d and d['<UNK>'] == 2:
                return 2
            return 1

        self.source_unk_vals = [determine_unk_val(d)
                                for d in self.source_dicts]
        self.target_unk_val = determine_unk_val(self.target_dict)

        self.target_graph = target_graph
        self.semantic_graph = target_semantic_graph
        self.remove_parse = remove_parse
        self.keep_data_in_memory = keep_data_in_memory
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.source_vocab_sizes = source_vocab_sizes
        self.target_vocab_size = target_vocab_size

        self.token_batch_size = token_batch_size

        if self.source_vocab_sizes != None:
            assert len(self.source_vocab_sizes) == len(self.source_dicts)
            for d, vocab_size in zip(self.source_dicts, self.source_vocab_sizes):
                if vocab_size != None and vocab_size > 0:
                    for key, idx in list(d.items()):
                        if idx >= vocab_size:
                            del d[key]

        if self.target_vocab_size != None and self.target_vocab_size > 0:
            for key, idx in list(self.target_dict.items()):
                if idx >= self.target_vocab_size:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def set_remove_parse(self, bool):
        self.remove_parse = bool

    def reset(self):
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
            if self.keep_data_in_memory:
                self.source, self.target = FileWrapper(self.source_orig, self.lines_file_path),\
                                           FileWrapper(self.target_orig, self.lines_file_path)
            else:
                self.source = fopen(self.source_orig, 'r')
                self.target = fopen(self.target_orig, 'r')
        if self.shuffle:
            if self.keep_data_in_memory:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
            else:
                self.source, self.target = shuffle.jointly_shuffle_files(
                    [self.source_orig, self.target_orig],
                    self.lines_file_path,
                    temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        longest_source = 0
        longest_target = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(
            self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()

                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                if self.remove_parse:
                    ss = extract_text_from_combined_tokens(ss)
                    tt = extract_text_from_combined_tokens(tt)
                if (ss and tt) or not self.ignore_empty:
                    self.source_buffer.append(ss)
                    self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by source/target buffer length
            if self.sort_by_length:
                tlen = numpy.array([max(len(s), len(t)) for (
                    s, t) in zip(self.source_buffer, self.target_buffer)])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        def lookup_token(t, d, unk_val):
            return d[t] if t in d else unk_val

        try:
            # actual work here
            while True:

                # read from source file and map to words index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [lookup_token(f, self.source_dicts[i],
                                          self.source_unk_vals[i])
                             for (i, f) in enumerate(w.split('|'))]
                    else:
                        w = [lookup_token(w, self.source_dicts[0],
                                          self.source_unk_vals[0])]
                    tmp.append(w)
                ss_indices = tmp
                source.append(ss_indices)

                # read from source file and map to words index
                tt = self.target_buffer.pop()

                tt_indices = [lookup_token(w, self.target_dict,
                                           self.target_unk_val) for w in tt]
                if self.target_vocab_size != None:
                    tt_indices = [w if w < self.target_vocab_size
                                  else self.target_unk_val
                                  for w in tt_indices]

                if self.target_graph:
                    # logging.info("read" + str(tt))
                    if self.semantic_graph:
                        logging.info("read" + str(tt))
                        tt_edge_time, tt_label_time, parents_time = convert_semantic_transitions_text_to_graph(
                                                                    ["<GO>"] + tt,
                                                                    self.maxlen + 1,
                                                                    self.target_labels,
                                                                    self.target_labels_num)
                        target.append((tt_indices, tt_edge_time, tt_label_time, parents_time))
                    else:
                        tt_edge_time, tt_label_time, parents_time = convert_text_to_graph(
                            ["<GO>"] + tt, self.maxlen + 1, self.target_labels, self.target_labels_num,
                            split=self.splitted_action)
                        target.append((tt_indices, tt_edge_time, tt_label_time, parents_time))
                else:
                    target.append(tt_indices)

                longest_source = max(longest_source, len(ss_indices))
                longest_target = max(longest_target, len(tt_indices))

                if self.token_batch_size:
                    if len(source) * longest_source > self.token_batch_size or \
                            len(target) * longest_target > self.token_batch_size:
                        # remove last sentence pair (that made batch over-long)
                        source.pop()
                        target.pop()
                        self.source_buffer.append(ss)
                        self.target_buffer.append(tt)

                        break

                else:
                    if len(source) >= self.batch_size or \
                            len(target) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True
        return source, target


def _last_word(idxs, tokens,
               graceful=False):
    if not idxs:
        if not graceful:
            raise IndexError("Asked for last words on an empty buffer")
        return [], []
    i = -1
    while -i < len(idxs):
        i -= 1
        if not tokens[i].endswith("@@"):
            i += 1
            break
    return idxs[i:], tokens[i:]


def convert_text_to_graph(x_target, max_len, labels_dict, num_labels, split=False, attend_max=False, graceful=False):
    # TODO document
    # TODO connect reduce and labels to the popped out
    # TODO convert to work with indexes
    slf = 0
    lft = 1
    right = 2
    # TODO try results with separate edge type
    lft_edge = lft
    right_edge = right
    edge_types_num = 3
    edge_times = numpy.zeros((max_len, max_len, edge_types_num)) + float("inf")
    for i in range(edge_times.shape[0]):
        edge_times[i, i, slf] = i
    if attend_max:
        sentence_len = max_len
    else:
        sentence_len = len(x_target)
    parents = numpy.zeros((sentence_len, sentence_len), dtype=np.float32) + float("inf")
    for i in range(parents.shape[0]):
        parents[i, i] = i
    if num_labels:
        label_times = numpy.zeros((max_len, max_len, num_labels)) + float("inf")
    token_num = 0
    idxs_stack = []
    tokens_stack = []
    label = False
    root = False
    edge_end = "@@|"
    label_start = "|@@"
    # if ("L@@|" not in x_target):
    #     raise
    for token_id, token in enumerate(x_target):
        # last = tokens_stack

        if token.endswith(edge_end):
            assert graceful or not label or not num_labels, token + "," + " ".join(x_target)
            label = True
            if split:
                raise ValueError("why")
            if token.startswith(ConllSent.REDUCE_L):
                min_len_cond = len(idxs_stack) > 1
                if graceful and not min_len_cond:
                    heads_ids = []
                    continue
                assert min_len_cond, "tried to create a left edge with 1 words or less in the buffer " + x_target
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(heads_ids)], tokens_stack[:-len(heads_ids)]
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                idxs_stack += heads_ids
                tokens_stack += head_tokens
            elif token.startswith(ConllSent.REDUCE_R):
                if split:
                    label_token = x_target[token_id + 1]
                else:
                    label_token = token
                    assert label_token == x_target[token_id], (label_token, token_id, x_target)
                root = graceful or "root" in label_token  # Root expects only one words to exist
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=root)
            else:
                raise ValueError("Unexpected reduce action token" + token)

            # add parents
            for head in heads_ids:
                parents[token_id, head] = token_id
                for dependent in dependent_ids:
                    parents[token_id, dependent] = token_id
                    parents[dependent, head] = token_id
                    parents[dependent, token_id] = token_id

            # add edge
            for head in heads_ids:
                for dependent in dependent_ids:
                    edge_times[head, dependent, lft] = token_id
                    edge_times[dependent, head, right] = token_id

            # add a label to and from edge tokens
            for head in heads_ids:
                edge_times[token_id, head, lft_edge] = token_id
            for dependent in dependent_ids:
                edge_times[token_id, dependent, right_edge] = token_id

            if split:
                assert token.endswith(edge_end)
            elif num_labels:
                if token not in labels_dict or labels_dict[token] >= label_times.shape[-1]:
                    logging.error(
                        f"Token error, token: {token} labels_dict:{labels_dict} label_times shape {label_times.shape}")
                    logging.error("token num:", labels_dict[token])
                for head in heads_ids:
                    for dependent in dependent_ids:
                        label_times[head, dependent, labels_dict[token]] = token_id
                label = False

        elif token.startswith(label_start):
            assert not split
            assert graceful or label or not num_labels
            assert graceful or "root" not in token or root
            if label and num_labels:
                label = False
                if token not in labels_dict or labels_dict[token] >= label_times.shape[-1]:
                    logging.error(
                        f"Token error, token: {token} labels_dict:{labels_dict} label_times shape {label_times.shape}")
                    logging.error("token num:", labels_dict[token])
                for head in heads_ids:
                    for dependent in dependent_ids:
                        label_times[head, dependent, labels_dict[token]] = token_id

                # add label relevance to parents
                # add parents
                for head in heads_ids:
                    parents[token_id, head] = token_id
                    for dependent in dependent_ids:
                        parents[token_id, dependent] = token_id
                        parents[dependent, token_id] = token_id

                # add a label to and from edges
                for head in heads_ids:
                    edge_times[token_id, head, lft_edge] = token_id
                for dependent in dependent_ids:
                    edge_times[token_id, dependent, right_edge] = token_id

        else:
            assert graceful or not label or not num_labels, token + "," + " ".join(x_target)
            label = False
            idxs_stack.append(token_id)
            tokens_stack.append(token)
            token_num += 1  # number of tokens that are not transitions (actual subwords)
    # print("edge array", np.array(edge_times, dtype=np.float32))
    # print("x_target for graph convert", x_target, np.array(x_target).shape)

    edge_times = np.array(edge_times, dtype=np.float32)
    if num_labels:
        label_times = np.array(label_times, dtype=np.float32)
    else:
        label_times = np.zeros(1, dtype=np.float32)
    # print("initial parents", parents, parents.shape)
    # print("initial parents non_inf", np.argwhere(parents < 1000), parents[parents < 1000])
    # print("initial shapes", edge_times.shape, label_times.shape, parents.shape)
    return edge_times, label_times, parents


def convert_semantic_transitions_text_to_graph(x_target,
                                           max_len,
                                           labels_dict,
                                           num_labels,
                                           attend_max=False,
                                           graceful=False,
                                            split=False):
    """

    Must send x_target with <GO> at beggining of each line, and max_len+1
    :param x_target:    target transitions set.
    :param max_len:     max sentence len
    :param labels_dict: dict mapping labels to an index (used to create the labels matrix)
    :param num_labels:  number of available labels
    :param attend_max:  whether to use the max_len parameter
    :param graceful:    whether to abort when there's an error
    :param split:       where to split the edges and their labels (NOT SUPPORTED!)
    :return:            Matrics of the edges and the labels.
    """
    assert not split # not supporting split labels
    print(f"parsing transitions line to graph.")
    slf = 0
    lft = 1
    right = 2
    lft_edge = lft
    right_edge = right
    edge_types_num = 3
    edge_times = numpy.zeros((max_len, max_len, edge_types_num)) + float("inf")
    for i in range(edge_times.shape[0]):
        edge_times[i, i, slf] = i
    if attend_max:
        sentence_len = max_len
    else:
        sentence_len = len(x_target)
    parents = numpy.zeros((sentence_len, sentence_len), dtype=np.float32) + float("inf")
    for i in range(parents.shape[0]):
        parents[i, i] = i
    if num_labels:
        label_times = numpy.zeros((max_len, max_len, num_labels)) + float("inf")
    else:
        label_times = None
    token_num = 0
    nodes_num = 0

    idxs_stack = []
    tokens_stack = []
    idxs_buffer = []  # buffer is RTL STACK
    tokens_buffer = []

    node_act_mark = "##|"
    edge_mark = "@@|"  # remote edges will be treated as edges with a new label ("remote-A")
    stack_action_mark = "$$|"
    node_label_idx = len("NODE-")
    # remote_edge_mark = "|^^"
    # root_mark = "|**"

    for token_id, token in enumerate(x_target):
        if token_id == 0:  # root is added to the word at the beginning, add it to stack
            idxs_stack.append(token_id)
            tokens_stack.append(token)
        elif token.endswith(node_act_mark):
            # creating a node in the buffer - each node will get an idx that will identify it (node_i)
            # Creating a node also adds a left edge to it (from node to top word in stack)
            idxs_buffer.append(token_id)
            tokens_buffer.append(f"node_{nodes_num}")
            nodes_num += 1
            node_label = token[node_label_idx]
            edge_label_token = "R-EDGE-" + node_label + edge_mark
            dependent_ids, _ = _last_word(idxs_stack, tokens_stack, graceful=graceful)
            heads_ids = [token_id]

            add_edges(dependent_ids, edge_times, heads_ids, label_times, labels_dict, lft,
                  lft_edge, num_labels, parents, right, right_edge, edge_label_token, token_id, is_node=True)
        elif token.endswith(stack_action_mark):
            if token.startswith("REDUCE"):  # remove top word from the stack

                last_word_ids, last_word_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(last_word_ids)], \
                                           tokens_stack[:-len(last_word_tokens)]
            elif token.startswith("SHIFT"):  # move from buffer to stack - used for nodes and swapped words
                pop_buffer_ids, pop_buffer_tokens = _last_word(idxs_buffer, tokens_buffer, graceful=graceful)
                idxs_buffer, tokens_buffer = idxs_buffer[:-len(pop_buffer_ids)],\
                                             tokens_buffer[:-len(pop_buffer_tokens)]
                idxs_stack += pop_buffer_ids
                tokens_stack += pop_buffer_tokens
            elif token.startswith("SWAP"):  # move 2nd word in stack from stack back to buffer
                # need to keep last words in stack - save them in vars
                last_word_ids, last_word_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)

                idxs_stack, tokens_stack = idxs_stack[:-len(last_word_ids)], \
                                           tokens_stack[:-len(last_word_tokens)]
                # get words that will move to buffer (2nd to last words), and move them to buffer
                second_word_ids, second_word_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_buffer += second_word_ids
                tokens_buffer += second_word_tokens
                # remove 2nd to last words from stack
                idxs_stack, tokens_stack = idxs_stack[:-len(second_word_ids)], \
                                           tokens_stack[:-len(second_word_tokens)]
                # put back last words in stack
                idxs_stack += last_word_ids
                tokens_stack += last_word_tokens

        elif token.endswith(edge_mark):
            # assert graceful or not num_labels, token + "," + " ".join(x_target)
            assert not split
            if token.startswith("L"):
                min_len_cond = len(idxs_stack) > 1
                if graceful and not min_len_cond:
                    heads_ids = []
                    continue
                assert min_len_cond, "tried to create a left edge with 1 words or less in the buffer " + x_target

                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)

                no_head_idxs_stack, no_head_tokens_stack = idxs_stack[:-len(heads_ids)], tokens_stack[:-len(heads_ids)]
                dependent_ids, dependent_tokens = _last_word(no_head_idxs_stack, no_head_tokens_stack, graceful=graceful)

            elif token.startswith("R"):
                label_token = token
                assert label_token == x_target[token_id], (label_token, token_id, x_target)
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                no_dep_idxs_stack, no_dep_tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_tokens)]
                heads_ids, head_tokens = _last_word(no_dep_idxs_stack, no_dep_tokens_stack, graceful=graceful)

            elif not token.startswith("FINISH"):
                raise ValueError("Unexpected action token" + token)

            add_edges(dependent_ids, edge_times, heads_ids, label_times, labels_dict, lft,
                              lft_edge, num_labels, parents, right, right_edge, token, token_id)

        else:  # word to add
            # assert graceful or not num_labels, token + "," + " ".join(x_target)
            idxs_stack.append(token_id)
            tokens_stack.append(token)
            token_num += 1  # number of tokens that are not transitions (actual subwords)

    edge_times = np.array(edge_times, dtype=np.float32)
    # print("edge array", edge_times)
    # print("edge array non_inf", np.argwhere(edge_times < 1000), edge_times[edge_times<1000])
    # print("x_target for graph convert", x_target, np.array(x_target).shape)
    # print(f"edge_times shape:{edge_times.shape}, label_times shape:{label_times.shape}, x_target len:{len(x_target)}\n")
    if num_labels:
        label_times = np.array(label_times, dtype=np.float32)
    else:
        label_times = np.zeros(1, dtype=np.float32)
    return edge_times, label_times, parents


def add_edges(dependent_ids, edge_times, heads_ids, label_times, labels_dict, lft, lft_edge,
              num_labels, parents, right, right_edge, token, token_id, is_node=False):
    # add parents
    for head in heads_ids:
        parents[token_id, head] = token_id
        for dependent in dependent_ids:
            parents[token_id, dependent] = token_id
            parents[dependent, head] = token_id
            parents[dependent, token_id] = token_id

    # add edge
    for head in heads_ids:
        for dependent in dependent_ids:
            edge_times[head, dependent, lft] = token_id
            edge_times[dependent, head, right] = token_id
    if not is_node:
        for head in heads_ids:
            edge_times[token_id, head, lft_edge] = token_id
        for dependent in dependent_ids:
            edge_times[token_id, dependent, right_edge] = token_id

    # add a label to and from edge tokens
    if num_labels:
        if token not in labels_dict or labels_dict[token] >= label_times.shape[-1]:
            logging.error(
                f"Token error, token: {token} labels_dict:{labels_dict} label_times shape {label_times.shape}")
            logging.error("token num:", labels_dict[token])
        for head in heads_ids:
            for dependent in dependent_ids:
                label_times[head, dependent, labels_dict[token]] = token_id

