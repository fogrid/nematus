#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:4,vmem:8g
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=oded.fogel@mail.huji.ac.il
# #SBATCH --output=/cs/snapless/oabend/fogrid/nematus/semantics/slurm/en-de_gcn%j.out


####################### Setup path and data Variables
script_dir=/cs/snapless/oabend/fogrid/nematus/semantics/scripts
echo "script_dir is ${script_dir}"
main_dir=$script_dir/..
echo "main_dir is ${main_dir}"
nematus_home=$main_dir/..
echo "nematus_home is ${nematus_home}"

data_dir=$main_dir/data

dev_prefix=newstest2012

trg_train_raw=$data_dir/${dev_prefix}.unesc.tok.tc.en
bpe_file=$data_dir/${dev_prefix}.unesc.tok.tc.bpe.en
ucca_input=${trg_train_raw}.txt
ucca_output_dir="${trg_train_raw}_ucca_res/"

trg_train=$data_dir/${dev_prefix}.unesc.tok.tc.en.ucca_trns


####################### prepeare UCCA transitiosn file
#if [ ! -f ${trg_train} ]; then
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
find . -maxdepth 1 -type f -name "${trg_train_raw}_*.xml" -exec mv -t $ucca_output_dir "{}" +
python ${main_dir}/semantic_transition_parsing.py "$ucca_output_dir" "$trg_train"  "$bpe_file"
#fi
