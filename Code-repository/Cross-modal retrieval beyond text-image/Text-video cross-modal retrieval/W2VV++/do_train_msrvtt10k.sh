
rootpath=$HOME/VisualSearch

#./do_train_msrvtt10k.sh w2vvpp_resnext101_subspace
#./do_train_msrvtt10k.sh w2vvpp_resnext101-resnet152_subspace

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 config"
    exit
fi


trainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
config=$1


gpu=1
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection \
    --rootpath $rootpath --config $config --model_prefix runs_0
