#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:4,vmem:8g
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=oded.fogel@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/fogrid/nematus/semantics/slurm/de-en_ucca_trns%j.out

####################### Setup path and data Variables
vocab_in=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.de
vocab_out=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.en
#script_dir=`dirname $0`
script_dir=/cs/snapless/oabend/fogrid/nematus/semantics/scripts
main_dir=$script_dir/..
echo "script_dir is ${script_dir}"
echo "main_dir is ${main_dir}"
nematus_home=$main_dir/..
echo "nematus_home is ${nematus_home}"

data_dir=$main_dir/data
model_dir=$script_dir/models
mkdir -p $model_dir
. $script_dir/vars

#working_dir=$model_dir/test
working_dir=$model_dir/prod
mkdir -p $working_dir

# Setup the files used for training
dev_prefix=train.clean

src_train=$data_dir/${dev_prefix}.unesc.tok.tc.bpe.de

trg_train_raw=$data_dir/${dev_prefix}.unesc.tok.tc.en
bpe_file=$data_dir/${dev_prefix}.unesc.tok.tc.bpe.en
ucca_input=${trg_train_raw}.txt
ucca_output_dir="${trg_train_raw}_ucca_res/"

trg_train=$data_dir/${dev_prefix}.unesc.tok.tc.en.ucca_trns

# Setup the files used for validation.
valid_prefix=newstest2013
src_valid=$data_dir/${valid_prefix}.unesc.tok.tc.bpe.de
trg_valid=$data_dir/${valid_prefix}.unesc.tok.tc.en.ucca_trns

####################### prepeare UCCA transitiosn file
if [ ! -f ${trg_train} ]; then
  echo "creating target file ${trg_train}"
  if [ ! -f ${ucca_input} ]; then
    echo "creating input to ucca ${ucca_input}"
    # add .txt to the file (ucca requires this)
    cp $trg_train_raw $ucca_input
    # separate each sentence with empty line (by replacing single \n with double \n)
    perl -i -pe 's/\n/\n\n/g' $ucca_input
  fi

  python -m tupa "$ucca_input" --lang en -m bert_multilingual_layers_4_layers_pooling_weighted_align_sum
  mkdir --parents "$ucca_output_dir"
  find . -maxdepth 1 -type f -name "${trg_train_raw }_*.xml" -exec mv -t $ucca_output_dir "{}" +
  python ${main_dir}/semantic_transition_parsing.py "$ucca_output_dir" "$trg_train"  "$bpe_file"
fi

####################### Train model

src_bpe=$src_train.json
trg_bpe=$trg_train.json

# create dictionary if needed
if [ ! -f ${trg_bpe} ]; then
    echo "creating target dict"
    tmp="$working_dir/tmp_all_train"
    cat $vocab_out $trg_train $trg_valid> $tmp
    python "${nematus_home}/data/build_dictionary.py" $tmp
    mv "$tmp.json" $trg_bpe
    rm $tmp
fi

if [ ! -f ${src_bpe} ]; then
    echo "creating source dict"
    tmp="$working_dir/tmp_all_train"
    cat $vocab_in $src_train $src_valid > $tmp
    python "${nematus_home}/data/build_dictionary.py" $tmp
    mv "$tmp.json" $src_bpe
    rm $tmp
fi

len=160
batch_size=128
embedding_size=256
# token_batch_size=2048
# sent_per_device=4
tokens_per_device=100
dec_blocks=4
enc_blocks="${dec_blocks}"
lshw -C display | tail # write the acquired gpu properties

python "${nematus_home}/nematus/train.py" \
    --source_dataset $src_train \
    --target_dataset $trg_train \
    --dictionaries $src_bpe $trg_bpe\
    --save_freq 1000 \
    --model $working_dir/model_seq_trans.npz \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size $embedding_size \
    --state_size $embedding_size \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --transformer_dec_depth $dec_blocks \
    --transformer_enc_depth $enc_blocks \
    --learning_schedule transformer \
    --warmup_steps 4000 \
    --maxlen $len \
    --batch_size $batch_size \
    --disp_freq 100 \
    --sample_freq 0 \
    --beam_freq 1000 \
    --translation_maxlen $len \
    --beam_size 8 \
    --target_labels_num 45\
    --non_sequential \
    --target_graph \
    --normalization_alpha 0.6\
    --valid_source_dataset $src_valid \
    --valid_target_dataset $trg_valid \
    --valid_batch_size 4 \
    --max_tokens_per_device $tokens_per_device \
    --valid_freq 10000 \
    --valid_script "${script_dir}/validate_seq.sh" \
    --valid_remove_parse \
    --target_semantic_graph \
    --lines_file ${trg_train}.line_nums \
    --valid_lines_file $trg_valid.line_nums


echo done