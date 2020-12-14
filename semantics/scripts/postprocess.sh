#!/bin/bash

script_dir=/cs/snapless/oabend/fogrid/nematus/semantics/scripts

#model_dir=`dirname $0`
model_dir=$script_dir/models

#language-independent variables (toolkit locations)
. $script_dir/vars
trg=en

sed 's/\@\@ //g' | \
$moses_scripts/recaser/detruecase.perl | \
$moses_scripts/tokenizer/detokenizer.perl -l $trg
