CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
--master_addr 127.0.0.9 --master_port 29509 main_task_retrieval.py \
--do_train --num_thread_reader=8 --epochs=5 --batch_size=64 --n_display=20 \
--train_csv PATH_TO_MARVTT/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv PATH_TO_MARVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path PATH_TO_MARVTT/msrvtt_data/MSRVTT_data.json \
--features_path PATH_TO_MARVTT/MSRVTT/frames_30fps \
--output_dir PATH_TO_SAVE_CHECKPOINTS \
--lr 5e-5 --max_words 32 --max_frames 12 --batch_size_val 8 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--strategy 2 \
--pretrained_clip_name ViT-B/16 \
--extra_video_cls_num 2 \
--extra_text_cls_num 2 \
--n_video_embeddings 7 \
--n_text_embeddings 7