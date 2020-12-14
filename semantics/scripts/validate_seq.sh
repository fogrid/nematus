#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the newstest2013 test set
# using detokenized BLEU (equivalent to evaluation with mteval-v13a.pl).

translations=$1

script_dir=/cs/snapless/oabend/fogrid/nematus/semantics/scripts
main_dir=$script_dir/../
nematus_home=$script_dir/../
data_dir=$main_dir/data

#language-independent variables (toolkit locations)
. $script_dir/vars

#language-dependent variables (source and target language)
. $script_dir/vars
remove_edges=$nematus_home/nematus/parsing/remove_edges.py

dev_prefix=newstest2013
src_dev=$data_dir/${dev_prefix}.unesc.tok.tc.bpe.de
trg_dev=$data_dir/${dev_prefix}.unesc.tok.tc.en.ucca_trns

trg=en

ref=$data_dir/$dev_prefix.ref.$trg

# create ref file if needed
if [ ! -f $ref ] ; then
	trns=$data_dir/$dev_prefix.$trg.ucca_trns
	if [ ! -f $trns ] ; then
		$script_dir/postprocess.sh < "$data_dir/$dev_prefix.unesc.tok.tc.$trg.ucca_trns" > "$trns"
	fi
	python $remove_edges $trns -o $ref
fi

# write resulting file
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
$script_dir/postprocess.sh < "$translations" > "${script_dir}/output/out_${dev_prefix}_$current_time.$trg"

 
# evaluate translations and write BLEU score to standard output (for
# use by nmt.py)
tmp="tmp_postprocessed$current_time"
$script_dir/postprocess.sh < $translations > "$tmp.out" 
python $remove_edges "$tmp.out" |\
    $nematus_home/data/multi-bleu-detok.perl $ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
rm "$tmp.out"