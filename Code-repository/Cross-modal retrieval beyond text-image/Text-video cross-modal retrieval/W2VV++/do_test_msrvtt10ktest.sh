
rootpath=$HOME/VisualSearch
overwrite=0

trainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
#config=w2vvpp_resnext101-resnet152_subspace
#model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/runs_0/model_best.pth.tar
#sim_name=$trainCollection/$valCollection/$config

#./do_test_msrvtt10ktest.sh ~/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/w2vvpp_resnext101-resnet152_subspace/runs_0/model_best.pth.tar msrvtt10ktrain/msrvtt10kval/w2vvpp_resnext101-resnet152_subspace/runs_0

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 model_path sim_name"
    exit
fi

testCollection=msrvtt10ktest
model_path=$1
sim_name=$2

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection $model_path $sim_name \
    --query_sets $testCollection.caption.txt \
    --rootpath $rootpath  --overwrite $overwrite
