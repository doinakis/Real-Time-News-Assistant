# !/bin/bash
vevn_path='/home/doinakis/venv/Real-Time-News-Assistant/bin/python3'
xquad_file='/home/doinakis/github/haystack/xquad-dataset/xquad.el.json'
eval_script='../ReaderEvaluation.py'

declare -a models=("deepset/xlm-roberta-large-squad2"
                  "Danastos/triviaqa_bert_el"
                  "Danastos/squad_bert_el"
                  "Danastos/nq_bert_el"
                  "Danastos/qacombination_bert_el"
                  "Danastos/newsqa_bert_el")

for model in "${models[@]}"
do
  echo "Evaluation of $model"
  $vevn_path $eval_script --xquad_file $xquad_file --model $model
  echo "Done!"
done