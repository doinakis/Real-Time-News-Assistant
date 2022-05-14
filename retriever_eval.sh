# !/bin/bash
n=$1

# Check if number
re='^[0-9]+$'

if ! [[ $n =~ $re ]] ; then
  echo "error: Not a number" >&2; exit 1
fi
for top_k in $(eval echo "{1..$n}")
do
  /home/doinakis/venv/Real-Time-News-Assistant/bin/python3 RetrieverEvaluation.py --xquad_file /home/doinakis/github/haystack/xquad-dataset/xquad.el.json --top_k_retriever $top_k
done
