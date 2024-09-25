from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
from metrics import compute_metrics_together
import time
import argparse
from modules.cluster.fast_kmeans import batch_fast_kmedoids
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_multievent import MeRetriever
from modules.optimization import BertAdam

from util import parallel_apply_2, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=86400))

global logger


def get_args(description='Me-Retriever on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--loss', type=str, default='balanced', help='loss function used')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--alpha', type=float, default=1.0, help="The relative scale of v2t and t2v loss.")
    parser.add_argument('--dynamic_alpha', action='store_true', help="Dynamically adjust alpha for loss calculation.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--sim_lambda", default=0.0, type=float, help="The coefficient of added similarity term")
    parser.add_argument('--post_process', type=str, default='none', choices=['none', 'cluster'],
                        help="clustering over frames")
    parser.add_argument('--post_cluster_centroids', type=int, default=1, help='clustering frame length')

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf", "maxP"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument('--save_feature_path', type=str,
                        default=None,
                        help='Used to save the CLIP features')
    # cluster algorithms
    parser.add_argument('--cluster_algo', type=str, default='kmediods++',
                        choices=['kmediods++', 'pooling'],
                        help="The type of cluster algorithms.")

    parser.add_argument('--cluster_embedding', type=int, default=0,
                        help="Whether using cluser embedding or not.")

    parser.add_argument('--cluser_embed_from_clip', type=int, default=1,
                        help="Whether using CLIP pretrained positional embedding to initialize cluster embedding.")

    parser.add_argument('--cluster_frame_embedding', type=int, default=0,
                        help="Whether using cluser frame embedding or not.")

    parser.add_argument('--adaptive_cls', type=int, default=0,
                        help="Whether adaptive [CLASS] token fusion.")

    # parser.add_argument('--position_embed_first', type=int, default=0,
    # 						help="When clusttering, add position embedding first.")

    # parser.add_argument('--time_embed_frist', type=int, default=0,
    # 						help="When clustering, add frame embedding first.")

    parser.add_argument('--aggregation', type=str, default=None,
                        choices=['mean', 'None'],
                        help="When clustering, how to aggregate a cluster.")

    parser.add_argument('--cluster_iter_limit', type=int, default=100,
                        help="Iteration limits of cluster algorithms.")

    parser.add_argument('--cluster_distance', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="type of clustering distance.")

    parser.add_argument('--cluster_threshold', type=float, default=1e-5,
                        help="stop threshold for clustering.")

    parser.add_argument('--minkowski_norm_p', type=float, default=2.0,
                        help="p value for the p-norm distance to calculate between each vector pair.")

    # divide the pretrained temperature
    parser.add_argument('--temperature_new', type=float, default=1.0,
                        help='assign a new temperature to CLIP model')

    parser.add_argument('--time_embedding', type=int, default=0,
                        help="Add time embedding in CLIP model.")

    parser.add_argument('--pre_norm', type=int, default=0,
                        help="whether do l2 normalization before clustering.")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = MeRetriever.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                      task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    # model = apex.parallel.DistributedDataParallel(model)
    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
    }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = MeRetriever.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                          task_config=args)

        model.to(device)
    else:
        model = None
    return model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0.0
    total_loss1, total_loss2 = 0.0, 0.0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, group_mask, video, video_mask, vt_mask = batch
        if args.regularize == 'none':
            loss1, loss2, reg_loss = model(input_ids, input_mask, group_mask, video, video_mask, vt_mask)
            loss = loss1 + args.alpha * loss2
        else:
            loss1, loss2, reg_loss = model(input_ids, input_mask, group_mask, video, video_mask, vt_mask)
            loss = loss1 + args.alpha * loss2 + args.reg_lambda * reg_loss
        if torch.isnan(loss):
            loss1, loss2, reg_loss = model(input_ids, input_mask, group_mask, video, video_mask, vt_mask)
            print(loss1, loss2, reg_loss)
            raise ValueError
        
        if args.dynamic_alpha:
            args.alpha = loss1.item() / loss2.item()

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss1 += float(loss1)
        total_loss2 += float(loss2)
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, %f, Reg Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss1), args.alpha * float(loss2),
                            0,
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    sim_masks = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, group_mask = b1
        sequence_output, text_id = batch_sequence_output_list[idx1]
        each_row = []
        masks = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, vt_mask = b2
            visual_output = batch_visual_output_list[idx2]
            if model.multi2multi:
                b1b2_logits, sim_mask = model.get_similarity_sphere_eval(sequence_output, visual_output,
                                                                         video_mask, group_mask, text_id, idx2)
            else:
                b1b2_logits, sim_mask = model.get_similarity_logits(sequence_output, visual_output, input_mask,
                                                                    video_mask,
                                                                    group_mask, loose_type=model.loose_type)
                if text_id != idx2:
                    sim_mask = torch.zeros_like(b1b2_logits)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            sim_mask = sim_mask.cpu().detach().numpy()
            if b1b2_logits.shape[0] != sim_mask.shape[0] or b1b2_logits.shape[1] != sim_mask.shape[1]:
                print("ERROR: Shape inconsistency")
                raise AssertionError
            each_row.append(b1b2_logits)
            masks.append(sim_mask)
        each_row = np.concatenate(each_row, axis=-1)
        masks = np.concatenate(masks, axis=-1)
        sim_matrix.append(each_row)
        sim_masks.append(masks)
    return sim_matrix, sim_masks


def eval_epoch(args, model, test_dataloader, device, n_gpu):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    logger.info("model evaluation on {} GPU".format(n_gpu))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, group_mask, video, video_mask, vt_mask = batch
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            if model.cluster_inter:
                video_mask = model.get_video_mask_after_cluster(video_mask)
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, input_mask,
                                                                              video, video_mask, group_mask)
            if args.post_process == 'cluster':
                assign, medoids = batch_fast_kmedoids(visual_output, args.post_cluster_centroids,
                                                      distance=args.cluster_distance,
                                                      threshold=args.cluster_threshold,
                                                      iter_limit=args.cluster_iter_limit)
                idx = torch.arange(visual_output.shape[0], dtype=torch.long, device=visual_output.device).unsqueeze(-1)
                visual_output = visual_output[idx, medoids]
                video_mask = video_mask[idx, medoids]
                vt_mask = vt_mask[idx, :, medoids]
            elif args.post_process == 'perceiver':
                visual_output = model.perceiver_sampler(visual_output, video_mask)
                video_mask = torch.ones(visual_output.shape[0], visual_output.shape[1]).to(visual_output.device)

            batch_sequence_output_list.append((sequence_output, bid))
            batch_list_t.append((input_mask, group_mask,))

            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask, vt_mask))

            if (bid + 1) % args.n_display == 0:
                logger.info("Evaluation step: {}/{}".format(bid, len(test_dataloader)))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [(b[0].to(devc), b[1]) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[did], batch_list_v_splits[did],
                                      batch_t_output_splits[did], batch_v_output_splits[did]) for did in device_ids]
            parallel_outputs, parallel_mask = parallel_apply_2(_run_on_single_gpu, model, parameters_tuple_list,
                                                               device_ids)
            sim_matrix, sim_matrix_mask = [], []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
                sim_matrix_mask += parallel_mask[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_mask = np.concatenate(sim_matrix_mask, axis=0)
        else:
            sim_matrix, sim_matrix_mask = _run_on_single_gpu(model, batch_list_t, batch_list_v,
                                                             batch_sequence_output_list,
                                                             batch_visual_output_list)
            sim_matrix = np.concatenate(sim_matrix, axis=0)
            sim_matrix_mask = np.concatenate(sim_matrix_mask, axis=0)

    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics, vt_metrics = compute_metrics_together(sim_matrix.T, sim_matrix_mask.T)
    logger.info("Text to video:")
    for key in tv_metrics:
        logger.info("{}: {}".format(key, tv_metrics[key]))
    logger.info("Video to text:")
    for key in vt_metrics:
        logger.info("{}: {}".format(key, vt_metrics[key]))


@record
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                     args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model not in [None, 'None']:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch'] + 1
            resumed_loss = checkpoint['loss']

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                eval_epoch(args, model, test_dataloader, device, n_gpu)

        ## Uncomment if you want to test on the best checkpoint
        # if args.local_rank == 0:
            #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            # eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
