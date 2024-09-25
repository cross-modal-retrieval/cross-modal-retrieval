
rootpath=$HOME/VisualSearch

#./do_train.sh tgif-msrvtt10k tv2016train setA w2vvpp_resnext101_subspace
#./do_train.sh tgif-msrvtt10k tv2016train setA w2vvpp_resnext101-resnet152_subspace

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 trainCollection valCollection val_set config"
    exit
fi


trainCollection=$1 #tgif-msrvtt10k
valCollection=$2 #tv2016train
val_set=$3 #setA
config=$4

python trainer.py $trainCollection $valCollection \
    --rootpath $rootpath --config $config --val_set $val_set --model_prefix runs_0
