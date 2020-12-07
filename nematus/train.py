#!/usr/bin/env python3
'''
Build a neural machine translation model with soft attention
'''
import collections
from datetime import datetime
import json
from shutil import copyfile
import os
import locale
import logging
import subprocess
import sys
import tempfile
import time

# Start logging.
level = logging.INFO
logging.basicConfig(level=level, format='%(levelname)s: %(message)s',  datefmt="%m-%d %H:%M")

import numpy as np
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .beam_search_sampler import BeamSearchSampler
    from .config import read_config_from_cmdline, write_config_to_json_file
    from .data_iterator import TextIterator
    from .exponential_smoothing import ExponentialSmoothing
    from . import learning_schedule
    from . import model_loader
    from .model_updater import ModelUpdater
    from .random_sampler import RandomSampler
    from . import rnn_model
    from . import tf_utils
    from .transformer import Transformer as TransformerModel
    from . import translate_utils
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    from beam_search_sampler import BeamSearchSampler
    from config import read_config_from_cmdline, write_config_to_json_file
    from data_iterator import TextIterator
    from exponential_smoothing import ExponentialSmoothing
    import learning_schedule
    import model_loader
    from model_updater import ModelUpdater
    from random_sampler import RandomSampler
    import rnn_model
    import tf_utils
    from transformer import Transformer as TransformerModel
    import translate_utils
    import util

np.random.seed(0)
tf.random.set_seed(0)


def load_data(config):
    logging.info(f'Reading data...')
    logging.info(f"source {config.source_dataset}")
    logging.info(f"target {config.target_dataset}")

    logging.info(f"source dict {config.source_dicts}")
    logging.info(f"target dict {config.target_dict}")
    text_iterator = TextIterator(
        source=config.source_dataset,
        target=config.target_dataset,
        source_dicts=config.source_dicts,
        target_dict=config.target_dict,
        model_type=config.model_type,
        batch_size=config.batch_size,
        maxlen=config.maxlen,
        source_vocab_sizes=config.source_vocab_sizes,
        target_vocab_size=config.target_vocab_size,
        skip_empty=True,
        shuffle_each_epoch=config.shuffle_each_epoch,
        sort_by_length=config.sort_by_length,
        use_factor=(config.factors > 1),
        maxibatch_size=config.maxibatch_size,
        token_batch_size=config.token_batch_size,
        keep_data_in_memory=config.keep_train_set_in_memory,
        preprocess_script=config.preprocess_script,
        target_graph=config.target_graph,
        target_semantic_graph=config.target_semantic_graph,
        target_labels_num=config.target_labels_num,
        splitted_action=config.split_transitions,
        lines_file_path=config.lines_file
    )

    if config.valid_freq and config.valid_source_dataset and config.valid_target_dataset:

        remove_parse = True if config.valid_remove_parse else False
        valid_text_iterator = TextIterator(
            source=config.valid_source_dataset,
            target=config.valid_target_dataset,
            source_dicts=config.source_dicts,
            target_dict=config.target_dict,
            model_type=config.model_type,
            batch_size=config.valid_batch_size,
            maxlen=config.maxlen,
            source_vocab_sizes=config.source_vocab_sizes,
            target_vocab_size=config.target_vocab_size,
            shuffle_each_epoch=False,
            sort_by_length=True,
            use_factor=(config.factors > 1),
            maxibatch_size=config.maxibatch_size,
            token_batch_size=config.valid_token_batch_size,
            remove_parse=remove_parse,
            target_graph=config.target_graph,
            target_semantic_graph=config.target_semantic_graph,
            target_labels_num=config.target_labels_num,
            splitted_action=config.split_transitions,
            ignore_empty=True,
            lines_file_path=config.valid_lines_file
        )
    else:
        logging.info('no validation set loaded')
        valid_text_iterator = None
    logging.info('Done')
    return text_iterator, valid_text_iterator


def train(config, sess):
    assert (config.prior_model != None and (
        tf.compat.v1.train.checkpoint_exists(os.path.abspath(config.prior_model))) or (config.map_decay_c == 0.0)), \
        "MAP training requires a prior model file: Use command-line option --prior_model"

    # Construct the graph, with one model replica per GPU

    num_gpus = len(tf_utils.get_available_gpus())
    num_replicas = max(1, num_gpus)

    if config.loss_function == 'MRT':
        assert config.gradient_aggregation_steps == 1
        assert config.max_sentences_per_device == 0, "MRT mode does not support sentence-based split"
        if config.max_tokens_per_device != 0:
            assert (
                        config.samplesN * config.maxlen <= config.max_tokens_per_device), "need to make sure candidates of a sentence could be " \
                                                                                          "feed into the model"
        else:
            assert num_replicas == 1, "MRT mode does not support sentence-based split"
            assert (
                        config.samplesN * config.maxlen <= config.token_batch_size), "need to make sure candidates of a sentence could be " \
                                                                                     "feed into the model"

    logging.info('Building model...')
    replicas = []
    for i in range(num_replicas):
        device_type = "GPU" if num_gpus > 0 else "CPU"
        device_spec = tf.DeviceSpec(device_type=device_type, device_index=i)
        with tf.device(device_spec):
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=(i > 0)):
                if config.model_type == "transformer":
                    model = TransformerModel(config)
                else:
                    model = rnn_model.RNNModel(config)
                replicas.append(model)

    init = tf.zeros_initializer()
    global_step = tf.compat.v1.get_variable('time', [], initializer=init, trainable=False)

    if config.learning_schedule == "constant":
        schedule = learning_schedule.ConstantSchedule(config.learning_rate)
    elif config.learning_schedule == "transformer":
        schedule = learning_schedule.TransformerSchedule(
            global_step=global_step,
            dim=config.state_size,
            warmup_steps=config.warmup_steps)
    elif config.learning_schedule == "warmup-plateau-decay":
        schedule = learning_schedule.WarmupPlateauDecaySchedule(
            global_step=global_step,
            peak_learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            plateau_steps=config.plateau_steps)
    else:
        logging.error('Learning schedule type is not valid: {}'.format(
            config.learning_schedule))
        sys.exit(1)

    if config.optimizer == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=schedule.learning_rate,
                                                     beta1=config.adam_beta1,
                                                     beta2=config.adam_beta2,
                                                     epsilon=config.adam_epsilon)
    else:
        logging.error(
            'No valid optimizer defined: {}'.format(config.optimizer))
        sys.exit(1)

    if config.summary_freq:
        summary_dir = (config.summary_dir if config.summary_dir is not None
                       else os.path.abspath(os.path.dirname(config.saveto)))
        writer = tf.compat.v1.summary.FileWriter(summary_dir, sess.graph)
    else:
        writer = None

    updater = ModelUpdater(config, num_gpus, replicas, optimizer, global_step,
                           writer)
    # val_updater = ModelUpdater(config, num_gpus, replicas, optimizer, global_step,
    #                        writer)

    if config.exponential_smoothing > 0.0:
        smoothing = ExponentialSmoothing(config.exponential_smoothing)

    saver, progress = model_loader.init_or_restore_variables(
        config, sess, train=True)

    global_step.assign(progress.uidx, sess)

    if config.sample_freq:
        random_sampler = RandomSampler(
            models=[replicas[0]],
            configs=[config],
            beam_size=1)

    if config.beam_freq or config.valid_script is not None:
        beam_search_sampler = BeamSearchSampler(
            models=[replicas[0]],
            configs=[config],
            beam_size=config.beam_size)

    # save model options
    write_config_to_json_file(config, config.saveto)

    text_iterator, valid_text_iterator = load_data(config)
    _, _, num_to_source, num_to_target = util.load_dictionaries(config)
    total_loss = 0.
    n_sents, n_words = 0, 0
    last_time = time.time()
    logging.info("Initial uidx={}".format(progress.uidx))
    logging.info("Number of trainable parameters:" + str(
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))

    # set epoch = 1 if print per-token-probability
    if config.print_per_token_pro:
        config.max_epochs = progress.eidx + 1
    for progress.eidx in range(progress.eidx, config.max_epochs):
        logging.info('Starting epoch {0} of {1}'.format(progress.eidx, config.max_epochs))
        for source_sents, target_sents in text_iterator:
            logging.info(f"Source len {len(source_sents)}")
            logging.info("Start batch {0}".format(progress.uidx))
            if len(source_sents[0][0]) != config.factors:
                logging.error(
                    'Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(
                        config.factors, len(source_sents[0][0])))
                sys.exit(1)
            if config.target_graph:
                target_sents, target_edges_time, target_labels_time, target_parents_time = list(zip(*target_sents))
                # # pad target sents to max_len so overall padding would occur (gcn does not allow dynamic sizes)
                # target_sents = [sent + [0] * (config.maxlen - 1 - len(sent)) for sent in target_sents]
                # source_sents = [sent + [[0]] * (config.maxlen - 1 - len(sent)) for sent in source_sents]
            else:
                target_edges_time = None
                target_labels_time = None
                target_parents_time = None
            logging.info("Predicting for " + str(len(target_sents)) + " sentences in batch.")
            # logging.info("Predicting for " + str(target_sents) + " sentences in batch.")
            # logging.info("Source sents " + str(source_sents) + " .")
            # logging.info("Target sents " + str(target_sents) + " .")
            x_in, x_mask_in, y_in, y_mask_in, target_edges_time, target_labels_time, target_parents_time = util.prepare_data(
                source_sents, target_sents, target_edges_time, target_labels_time, target_parents_time, config.factors,
                maxlen=None)

            if x_in is None:
                logging.info(
                    'Minibatch with zero sample under length {0}'.format(config.maxlen))
                continue
            write_summary_for_this_batch = config.summary_freq and ((progress.uidx % config.summary_freq == 0) or (
                    config.finish_after and progress.uidx % config.finish_after == 0))
            (factors, seqLen, batch_size) = x_in.shape

            output = updater.update(
                sess, x_in, x_mask_in, y_in, y_mask_in, num_to_target,
                write_summary_for_this_batch, target_edges_time, target_labels_time, target_parents_time)

            if config.print_per_token_pro == False:
                total_loss += output
            else:
                # write per-token probability into the file
                f = open(config.print_per_token_pro, 'a')
                for pro in output:
                    pro = str(pro) + '\n'
                    f.write(pro)
                f.close()

            n_sents += batch_size
            n_words += int(np.sum(y_mask_in))
            progress.uidx += 1

            # Update the smoothed version of the model variables.
            # To reduce the performance overhead, we only do this once every
            # N steps (the smoothing factor is adjusted accordingly).
            if config.exponential_smoothing > 0.0 and progress.uidx % smoothing.update_frequency == 0:
                sess.run(fetches=smoothing.update_ops)

            if config.disp_freq and progress.uidx % config.disp_freq == 0:
                duration = time.time() - last_time
                disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                logging.info('{0} Epoch: {1} Update: {2} Loss/words: {3} Words/sec: {4} Sents/sec: {5}'.format(
                    disp_time, progress.eidx, progress.uidx, total_loss / n_words, n_words / duration,
                                                             n_sents / duration))
                last_time = time.time()
                total_loss = 0.
                n_sents = 0
                n_words = 0

            if config.sample_freq and progress.uidx % config.sample_freq == 0:
                x_small = x_in[:, :, :10]
                x_mask_small = x_mask_in[:, :10]
                y_small = y_in[:, :10]
                samples = translate_utils.translate_batch(
                    sess, random_sampler, x_small, x_mask_small,
                    config.translation_maxlen, 0.0)
                assert len(samples) == len(x_small.T) == len(y_small.T), \
                    (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    # logging.info(f"Source origin all unk? {xx}, {num_to_source}")
                    source = util.factoredseq2words(xx, num_to_source)
                    target = util.seq2words(yy, num_to_target)
                    sample = util.seq2words(ss[0][0], num_to_target)
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    logging.info('SAMPLE: {}'.format(sample))

            if config.beam_freq and progress.uidx % config.beam_freq == 0:
                # logging.info(f"beam: x_in? {x_in}")
                x_small = x_in[:, :, :10]
                x_mask_small = x_mask_in[:, :10]
                y_small = y_in[:, :10]
                samples = translate_utils.translate_batch(
                    sess, beam_search_sampler, x_small, x_mask_small,
                    config.translation_maxlen, config.normalization_alpha)
                assert len(samples) == len(x_small.T) == len(y_small.T), \
                    (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    # logging.info(f"beam: Source origin all unk? {xx}, {num_to_source}")
                    source = util.factoredseq2words(xx, num_to_source)
                    target = util.seq2words(yy, num_to_target)
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    for i, (sample_seq, cost) in enumerate(ss):
                        sample = util.seq2words(sample_seq, num_to_target)
                        msg = 'SAMPLE {}: {} Cost/Len/Avg {}/{}/{}'.format(
                            i, sample, cost, len(sample), cost / len(sample))
                        logging.info(msg)
            logging.info("validation " + str(progress.uidx) + "," + str(config.valid_freq))
            if config.valid_freq and progress.uidx % config.valid_freq == 0:
                if config.exponential_smoothing > 0.0:
                    sess.run(fetches=smoothing.swap_ops)
                    valid_ce = validate(sess, replicas[0], config,
                                        valid_text_iterator, updater)
                    sess.run(fetches=smoothing.swap_ops)
                else:
                    valid_ce = validate(sess, replicas[0], config,
                                        valid_text_iterator, updater)
                logging.info("ce done")
                if (len(progress.history_errs) == 0 or
                        valid_ce < min(progress.history_errs)):
                    logging.info("ce improved over history_errs")
                    progress.history_errs.append(valid_ce)
                    progress.bad_counter = 0
                    save_non_checkpoint(sess, saver, config.saveto)
                    logging.info("saved to" + config.saveto)
                    progress_path = '{0}.progress.json'.format(config.saveto)
                    progress.save_to_json(progress_path)
                    logging.info("saved json")
                else:
                    progress.history_errs.append(valid_ce)
                    progress.bad_counter += 1
                    if progress.bad_counter > config.patience:
                        logging.info('Early Stop!')
                        progress.estop = True
                        break
                if config.valid_script is not None:
                    if config.exponential_smoothing > 0.0:
                        logging.info("validating with smoothing")
                        sess.run(fetches=smoothing.swap_ops)
                        score = validate_with_script(sess, beam_search_sampler)
                        sess.run(fetches=smoothing.swap_ops)
                    else:
                        logging.info("Validating without smoothing")
                        score = validate_with_script(sess, beam_search_sampler)
                    need_to_save = (score is not None and
                                    (len(progress.valid_script_scores) == 0 or
                                     score > max(progress.valid_script_scores)))
                    logging.info("validation done, saving?" + str(need_to_save))
                    if score is None:
                        score = 0.0  # ensure a valid value is written
                    progress.valid_script_scores.append(score)
                    if need_to_save:
                        progress.bad_counter = 0
                        save_path = config.saveto + ".best-valid-script"
                        save_non_checkpoint(sess, saver, save_path)
                        logging.info("saved to" + config.saveto)
                        write_config_to_json_file(config, save_path)
                        logging.info("saved json" + save_path)

                        progress_path = '{}.progress.json'.format(save_path)
                        logging.info("saved json to " + progress_path)
                        progress.save_to_json(progress_path)
                        logging.info("saved json")

            if config.save_freq and progress.uidx % config.save_freq == 0:
                logging.info("saving model")
                saver.save(sess, save_path=config.saveto,
                           global_step=progress.uidx)
                write_config_to_json_file(
                    config, "%s-%s" % (config.saveto, progress.uidx))

                progress_path = '{0}-{1}.progress.json'.format(
                    config.saveto, progress.uidx)
                progress.save_to_json(progress_path)

            if config.finish_after and progress.uidx % config.finish_after == 0:
                logging.info("Maximum number of updates reached")
                saver.save(sess, save_path=config.saveto,
                           global_step=progress.uidx)
                write_config_to_json_file(
                    config, "%s-%s" % (config.saveto, progress.uidx))

                progress.estop = True
                progress_path = '{0}-{1}.progress.json'.format(
                    config.saveto, progress.uidx)
                progress.save_to_json(progress_path)
                break
            logging.info("Loop done")
        if progress.estop:
            logging.info("Stopping")
            break
    logging.info("Finished training.")


def save_non_checkpoint(session, saver, save_path):
    """Saves the model to a temporary directory then moves it to save_path.

    Rationale: we use TensorFlow's standard tf.train.Saver mechanism for saving
    training checkpoints and also for saving the current best model according
    to validation metrics. Since these are all stored in the same directory,
    their paths would normally all get written to the same 'checkpoint' file,
    with the file containing whichever one was last saved. That creates a
    problem if training is interrupted after a best-so-far model is saved but
    before a regular checkpoint is saved, since Nematus will try to load the
    best-so-far model instead of the last checkpoint when it is restarted. To
    avoid this, we save the best-so-far models to a temporary directory, then
    move them to their desired location. The 'checkpoint' file that is written
    to the temporary directory can safely be deleted along with the directory.

    Args:
        session: a TensorFlow session.
        saver: a tf.train.Saver
        save_path: string containing the path to save the model to.

    Returns:
        None.
    """
    logging.info("Saving to " + save_path)
    head, tail = os.path.split(save_path)
    assert tail != ""
    base_dir = "." if head == "" else head
    with tempfile.TemporaryDirectory(dir=base_dir) as tmp_dir:
        tmp_save_path = os.path.join(tmp_dir, tail)
        logging.info("Saving temp to " + tmp_save_path)
        saver.save(session, save_path=tmp_save_path)
        for filename in os.listdir(tmp_dir):
            if filename == 'checkpoint':
                continue
            new = os.path.join(tmp_dir, filename)
            old = os.path.join(base_dir, filename)
            logging.info("Replacing " + new + " for " + old)
            os.replace(src=new, dst=old)
            logging.info("Replaced " + new + " for " + old)


def validate(session, model, config, text_iterator, updater):
    ce_vals, token_counts = calc_cross_entropy_per_sentence(
        session, model, config, text_iterator, updater=updater, normalization_alpha=0.0)
    num_sents = len(ce_vals)
    num_tokens = sum(token_counts)
    sum_ce = sum(ce_vals)
    avg_ce = sum_ce / num_sents
    logging.info('Validation cross entropy (AVG/SUM/N_SENTS/N_TOKENS): {0} '
                 '{1} {2} {3}'.format(avg_ce, sum_ce, num_sents, num_tokens))
    return avg_ce


def validate_with_script(session, beam_search_sampler):
    config = beam_search_sampler.configs[0]
    if config.valid_script == None:
        return None
    logging.info('Starting external validation.')
    out = tempfile.NamedTemporaryFile(mode='w')

    with open(config.valid_bleu_source_dataset, encoding="UTF-8") as infile:
        translate_utils.translate_file(
            input_file=infile,
            output_file=out,
            session=session,
            sampler=beam_search_sampler,
            config=config,
            max_translation_len=config.translation_maxlen,
            normalization_alpha=config.normalization_alpha,
            nbest=False,
            minibatch_size=config.valid_batch_size)
    logging.info("about to flush")
    out.flush()
    dev_out_path = os.path.splitext(config.saveto)[0] + "_val.out"
    logging.info("Script inputs: " + out.name + " " + dev_out_path)
    logging.info(
        "Saving dev transltion of " + config.valid_source_dataset + " to " + dev_out_path + " comparing to " + config.valid_target_dataset)
    copyfile(out.name, dev_out_path)

    args = [config.valid_script, out.name]
    proc = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout_bytes, stderr_bytes = proc.communicate()
    encoding = locale.getpreferredencoding()
    stdout = stdout_bytes.decode(encoding=encoding)
    stderr = stderr_bytes.decode(encoding=encoding)

    if len(stderr) > 0:
        logging.info("Validation script wrote the following to standard "
                     "error:\n" + stderr)
    if proc.returncode != 0:
        logging.warning("Validation script failed (returned exit status of "
                        "{}).".format(proc.returncode))
        return None
    try:
        score = float(stdout.split()[0])
    except:
        logging.warning("Validation script output does not look like a score: "
                        "{}".format(stdout))
        return None
    logging.info("Validation script score: {}".format(score))
    return score


def _aggregate_sentence_ce(ce_vals, token_counts):
    """
    Sums the cross-entropy values per sentence,
    notice that some values are thrown as they are added by the updater unnecessarily to prevent empty GPUs
    :param ce_vals:
    :param token_counts:
    :return:
    """
    # print("ce_vals", ce_vals)
    # print("token_counts", token_counts)
    ce_sums = []
    token_counts_idx = 0
    needed = token_counts[token_counts_idx]
    sentence_sum = 0
    for sub_batch in ce_vals:
        for replica in sub_batch:
            for token_ce in replica:
                sentence_sum += token_ce
                if sentence_sum == 0:  # skip trailing zeros from last sentence
                    continue
                needed -= 1
                if not needed:
                    token_counts_idx += 1
                    ce_sums.append(sentence_sum)
                    sentence_sum = 0
                    if token_counts_idx == len(token_counts):
                        return ce_sums
                    needed = token_counts[token_counts_idx]
                elif sentence_sum != 0 and token_ce == 0:
                    logging.warning("unexpected 0 in _aggregate_sentence_ce", token_ce)
            assert needed == token_counts[token_counts_idx], "sentence ces were gathered across gpus"

    assert needed == 0, f"Not enough cross-entropy values, expected {token_counts} sentence lengths"
    return ce_sums


def _flatten(nested):
    while isinstance(nested[0], (np.ndarray, list)) and len(nested) != 0:
        if isinstance(nested[0], (np.ndarray)):
            nested = np.concatenate(nested).ravel()
        if isinstance(nested[0], list):
            nested = [nest_item for item in nested for nest_item in item]
    return nested
    #
    # if (not isinstance(nested, list)) or nested == [] or len(nested) == 0:
    #     return nested
    # if isinstance(nested[0], list):
    #     return _flatten(nested[0]) + _flatten(nested[1:])
    # return nested[:1] + _flatten(nested[1:])


def calc_cross_entropy_per_sentence(session, model, config, text_iterator, updater,
                                    normalization_alpha=0.0):
    """Calculates cross entropy values for a parallel corpus.

    By default (when normalization_alpha is 0.0), the sentence-level cross
    entropy is calculated. If normalization_alpha is 1.0 then the per-token
    cross entropy is calculated. Other values of normalization_alpha may be
    useful if the cross entropy value will be used as a score for selecting
    between translation candidates (e.g. in reranking an n-nbest list). Using
    a different (empirically determined) alpha value can help correct a model
    bias toward too-short / too-long sentences.

    TODO Support for multiple GPUs

    Args:
        session: TensorFlow session.
        model: a RNNModel object.
        config: model config.
        text_iterator: TextIterator.
        normalization_alpha: length normalization hyperparameter.

    Returns:
        A pair of lists. The first contains the (possibly normalized) cross
        entropy value for each sentence pair. The second contains the
        target-side token count for each pair (including the terminating
        <EOS> symbol).
    """
    ce_vals, token_counts = [], []
    logging.info("calc_cross_entropy_per_sentence")
    text_iterator.set_remove_parse(False)
    for source_sents, target_sents in text_iterator:
        logging.info(f"Source len {len(source_sents)}")
        if not source_sents or not source_sents[0]:
            logging.error(f"Excepted source sents instead got: {source_sents}, target: {target_sents}, seen: {len(ce_vals)}")
            logging.info(f"source {config.valid_source_dataset}")
            logging.info(f"target {config.valid_target_dataset}")
            logging.info(f"source dict {config.source_dicts}")
            logging.info(f"target dict {config.target_dict}")
        if len(source_sents[0][0]) != config.factors:
            logging.error('Mismatch between number of factors in settings '
                          '({0}) and number present in data ({1})'.format(
                config.factors, len(source_sents[0][0])))
            sys.exit(1)
        if config.target_graph:
            target_sents, target_edges_time, target_labels_time, target_parents_time = list(zip(*target_sents))
            # # pad target sents to max_len so overall padding would occur (gcn does not allow dynamic sizes)
            # target_sents = [sent + [0] * (config.maxlen - 1 - len(sent)) for sent in target_sents]
            # source_sents = [sent + [[0]] * (config.maxlen - 1 - len(sent)) for sent in source_sents]
            logging.info("read sentence")
        else:
            target_edges_time = None
            target_labels_time = None
            target_parents_time = None

        x, x_mask, y, y_mask, x_edges_time, x_labels_time, x_parents_time = util.prepare_data(source_sents,
                                                                                              target_sents,
                                                                                              target_edges_time,
                                                                                             target_labels_time,
                                                                                              target_parents_time,
                                                                                              config.factors,
                                                                                              maxlen=None)

        # # Run the minibatch through the model to get the sentence-level cross entropy values.
        # feeds = {model.inputs.x: x,
        #          model.inputs.x_mask: x_mask,
        #          model.inputs.y: y,
        #          model.inputs.y_mask: y_mask,
        #          model.inputs.training: False}
        # if config.target_graph:
        #     timesteps = y.shape[0]
        #     feeds[model.inputs.edges] = util.times_to_input(x_edges_time, timesteps)
        #     if config.target_labels_num:
        #         feeds[model.inputs.labels] = util.times_to_input(x_labels_time, timesteps)
        # print("old op", model.loss_per_sentence )
        # run_options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)  # TODO delete
        # ce_vals = session.run(model.loss_per_sentence, feed_dict=feeds, options=run_options)

        batch_ce_vals = updater.loss_per_sentence(session, x, x_mask, y, y_mask, x_edges_time, x_labels_time, x_parents_time)

        # Optionally, do length normalization.
        batch_token_counts = [np.count_nonzero(s) for s in y_mask.T]
        if normalization_alpha:
            adjusted_lens = [
                n ** normalization_alpha for n in batch_token_counts]
            batch_ce_vals /= np.array(adjusted_lens)

        if config.target_graph or not config.sequential:
            # logging.info(f"gathering {np.array(batch_ce_vals).shape} {batch_ce_vals}")
            batch_ce_vals = _aggregate_sentence_ce(batch_ce_vals, batch_token_counts)

        # assert len(ce_vals) == len(token_counts), f"{len(ce_vals)} == {len(token_counts)}"
        # logging.info(f"batch_ce_vals {batch_ce_vals}")
        # logging.info(f"_flatten(batch_ce_vals) {_flatten(batch_ce_vals)}")
        batch_ce_vals = _flatten(batch_ce_vals)
        # if len(batch_ce_vals) and isinstance(batch_ce_vals[0], (np.ndarray, np.generic)):
        #     batch_ce_vals = np.concatenate(batch_ce_vals).ravel() # flatten lists and np arrays
        if len(batch_ce_vals) != len(batch_token_counts):
            if len(batch_ce_vals) > len(batch_token_counts):
                logging.warning(f"{len(batch_ce_vals)} == {len(batch_token_counts)} \n vals  {batch_ce_vals} token_counts\n {batch_token_counts}")
                logging.warning((f"Assumming unneeded values appear last: {batch_ce_vals[len(batch_token_counts):]}"))
                batch_ce_vals = batch_ce_vals[:len(batch_token_counts)]
            else:
                logging.warning(f"{len(batch_ce_vals)} == {len(batch_token_counts)} \n vals  {batch_ce_vals} token_counts\n {batch_token_counts}")
                raise ValueError("padding could result in more ce vals, the opposite should not happen")
        ce_vals += list(batch_ce_vals)
        token_counts += batch_token_counts
        logging.info("Seen {}".format(len(ce_vals)))
    # print("ce_vals", ce_vals)
    # print("token_counts", token_counts)
    assert len(ce_vals) == len(token_counts), f"{len(ce_vals)} == {len(token_counts)}"
    return ce_vals, token_counts


if __name__ == "__main__":
    import faulthandler

    # print where faults  happen (SIGSEGV, SIGFPE, SIGABRT, SIGBUS, and SIGILL signals)
    faulthandler.enable()

    # Parse command-line arguments.
    config = read_config_from_cmdline()
    logging.info(config)

    # TensorFlow 2.0 feature needed by ExponentialSmoothing.
    tf.compat.v1.enable_resource_variables()

    # Create the TensorFlow session.
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.allow_soft_placement = True

    # tf_config.gpu_options.allow_growth = True #TODO delete
    # print("allowing grouth")

    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # print("gpu fraction 0.95")

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    # memory_limit = 4000
    #         for gpu in gpus:
    #             tf.config.experimental.set_virtual_device_configuration(
    #                 gpu,
    #                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", "limited to", memory_limit)
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    # Train.
    with tf.compat.v1.Session(config=tf_config) as sess:
        train(config, sess)
