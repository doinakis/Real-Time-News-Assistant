# !/bin/bash
vevn_path='/home/doinakis/venv/Real-Time-News-Assistant/bin/python3'
xquad_file='/home/doinakis/github/haystack/xquad-dataset/xquad.el.json'
eval_script='RetrieverEvaluation.py'

# Check if input is nubmer
n=$1
re='^[0-9]+$'
if ! [[ $n =~ $re ]] ; then
  echo "error: Not a number" >&2; exit 1
fi

for top_k in $(eval echo "{1..$n}")
do
  $vevn_path $eval_script --xquad_file  $xquad_file --top_k_retriever $top_k
done
