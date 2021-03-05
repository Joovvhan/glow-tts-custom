config=$1
modeldir=$2

python multi_init.py -c $config -m $modeldir
python multi_train.py -c $config -m $modeldir
