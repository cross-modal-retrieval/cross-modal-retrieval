
train_collection='tgif-msrvtt10k'
overwrite=1

for encoding in bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding --overwrite $overwrite
done


