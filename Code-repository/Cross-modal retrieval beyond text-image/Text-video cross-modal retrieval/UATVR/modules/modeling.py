from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn, MILNCELoss, MILNCELoss_BoF, KLdivergence
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

try:
    from prob_models.pie_model import PIENet
    from prob_models.uncertainty_module import UncertaintyModuleImage
    from prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors
    from prob_models.probemb import MCSoftContrastiveLoss
except:
    raise EnvironmentError

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        # import pdb; pdb.set_trace()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        # import pdb; pdb.set_trace()

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs) # -----------
        

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]



class UATVR(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(UATVR, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
        
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.word_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)


        self.pie_net_video = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_video = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
        
        self.pie_net_text = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
        self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
        
        self.n_video_samples = self.task_config.n_video_embeddings        # numbers sampling from video distribution 
        self.n_text_samples = self.task_config.n_text_embeddings        # numbers sampling from text distribution

        # head
        self.text_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
            nn.Linear(transformer_width, 1))

        # loss function
        self.loss_fct = CrossEn()
        # self.loss_MIL_fct = MILNCELoss()    # Ex27-2 50.0
        # self.loss_mc_con = MCSoftContrastiveLoss(reduction='mean')
        self.loss_MIL_fct = MILNCELoss_BoF()
        self.vib_loss = KLdivergence()

        # extra class token num
        self.extra_cls_frame_num = self.task_config.extra_video_cls_num
        self.extra_cls_text_num = self.task_config.extra_text_cls_num

        show_log(task_config, "CLIP UATVR Model ......")
        show_log(task_config, "\t Extra video Class token number: {}".format(self.extra_cls_frame_num))
        show_log(task_config, "\t Extra text Class token number: {}".format(self.extra_cls_text_num))
        show_log(task_config, "\t Number of video sampling probabilistic embeddings: {}".format(self.n_video_samples))
        show_log(task_config, "\t Number of text sampling probabilistic embeddings: {}".format(self.n_text_samples))

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        # (B 1 32)  (B 1 132) (B 1 32)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape        # batch 2xclip num_frame 1 3 224 224
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, text_token = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            loss = 0.
            sim_matrix, mcsoft_loss, vib_loss = self.get_similarity_logits(sequence_output, text_token, visual_output, attention_mask, video_mask, 
                                                            shaped=True, loose_type=self.loose_type)

            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss    # token-wise DRL (extra v & t cls token)
            loss += mcsoft_loss  # superNCE
            loss += vib_loss    # kl divergence
            
            # show_log(self.task_config, "sim_loss:{}\t MILloss:{}\t vib_loss:{}".format(sim_loss, mcsoft_loss, vib_loss))

            return loss
        else:       # for inference
            return None
    
    def forward_eval(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        # (B 1 32)  (B 1 132) (B 1 32)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape        # batch 2xclip num_frame 1 3 224 224
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, text_token = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        sim_matrix, temp_logits = self.get_similarity_logits(sequence_output, text_token, visual_output, attention_mask, video_mask, 
                                                        shaped=True, loose_type=self.loose_type)

        return sim_matrix, temp_logits

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])     # B*num 32
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])  # B*num 32
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # B*num 32
        
        bs_pair = input_ids.size(0)     # B 2 32
        sequence_output, text_token = self.clip.encode_text(input_ids, return_hidden=True)     
        sequence_output = sequence_output.float()
        text_token = text_token.float()

        sequence_output = sequence_output.view(bs_pair, -1, sequence_output.size(-1))
        text_token = text_token.view(bs_pair, -1, text_token.size(-1))

        return sequence_output, text_token

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)        # video: bs*2*frame       video_frame:2*frame   video_mask:bs 2 frame
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            # video_mask = video_mask.view(-1, video_mask.shape[-1])        # bs 2 num_frame  =>  bs*2 num_frame

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, hidden_word = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
        return sequence_output, hidden_word, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, text_token, visual_output, attention_mask, video_mask, sim_header="seqTransf"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        frame_num = visual_output.size(1)   # 12 / 64
        word_num = text_token.size(1)       # 32 / 

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder +++++++++++++= extra token
            visual_output_original = visual_output

            extra_token_num = self.extra_cls_frame_num
            seq_length = visual_output.size(1) + extra_token_num     # extra 2 learnable token
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)    # bs num+extra_token_num dim
            frame_position_embeddings[:, 0:visual_output.size(1), :] += visual_output
            visual_output = frame_position_embeddings

            tempo_mask = torch.cat([video_mask, torch.ones(visual_output.size(0), extra_token_num).to(visual_output.device)], axis=1)
            extended_video_mask = (1.0 - tempo_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, tempo_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2).contiguous()  # LND -> NLD
            # multi extra token
            # visual_output = visual_output[:, :visual_output_original.size(1), :] + visual_output_original   # residual fusion
            visual_output[:, :visual_output_original.size(1), :] += visual_output_original
            video_mask = tempo_mask

            # sequential type: MLP for text with extra token
            text_original = text_token  # save original
            extra_text_num = self.extra_cls_text_num
            seq_text_length = extra_text_num + text_token.size(1)
            position_ids_text = torch.arange(seq_text_length, dtype=torch.long, device=text_token.device)
            position_ids_text = position_ids_text.unsqueeze(0).expand(text_token.size(0), -1)
            word_position_embeddings = self.word_position_embeddings(position_ids_text)
            word_position_embeddings[:, 0:text_token.size(1), :] += text_token
            text_token = word_position_embeddings

            tempo_mask_ = torch.cat([attention_mask, torch.ones(text_token.size(0), extra_text_num).to(text_token.device)], axis=1)
            extended_text_mask = (1.0 - tempo_mask_.unsqueeze(1)) * -1000000.0
            extended_text_mask = extended_text_mask.expand(-1, tempo_mask_.size(1), -1)
            text_token = text_token.permute(1, 0, 2)
            text_token = self.transformerClip(text_token, extended_text_mask)
            text_token = text_token.permute(1, 0, 2).contiguous()
            text_token[:, :text_original.size(1), :] += text_original
            attention_mask = tempo_mask_

        if self.training:       # works only on ddp
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)      # Bxgpu 1 dim
            text_token = allgather(text_token, self.task_config)        # Bxgpu 32 dim
            attention_mask = allgather(attention_mask, self.task_config)
            torch.distributed.barrier()
        
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_pooled = self._mean_pooling_for_similarity_visual(visual_output[:, 0:frame_num, :].contiguous(), video_mask[:, 0:frame_num].contiguous())
        visual_pooled = visual_pooled / visual_pooled.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        text_token = text_token / text_token.norm(dim=-1, keepdim=True)
        text_pooled = self._mean_pooling_for_similarity_sequence(text_token[:, 0:word_num, :].contiguous(), attention_mask[:, 0:word_num].contiguous())
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

        # ti_logits = self.token_wise_interaction(text_token=text_token, frame_token=visual_output, attention_mask=attention_mask, video_mask=video_mask)
        wti_logits = self.weighted_token_wise_intersection(text_token, visual_output, attention_mask, video_mask)
        
        ####################################################################
        ############### probabilistic embedding modeling part ##############
        ####################################################################
        prob_video = self.probabilistic_video(visual_pooled, visual_output[:, 0:frame_num, :].contiguous())
        prob_video_embedding = prob_video['embedding']      # 从分布中采样m个embedding
        prob_video_logsigma = prob_video['logsigma']         # 方差
        prob_video_mean = prob_video['mean']

        prob_text = self.probabilistic_text(text_pooled, text_token[:, 0:word_num, :].contiguous())
        prob_text_embedding = prob_text['embedding']       # b n 512
        prob_text_logsigma = prob_text['logsigma']   # bs 512
        prob_text_mean = prob_text['mean']       # bs 512
        
        if self.training:
            ####################################################################
            ############### MILNCELoss_(Supervised Contrastice Loss) ##############
            ####################################################################
            bs = prob_video_embedding.size(0)
            n_video = self.n_video_samples
            n_text = self.n_text_samples
            dim = prob_video_embedding.size(-1)

            prob_sim_matrix_from_v = torch.einsum('ad,bd->ab', [prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim)])
            MIL_loss_v = self.loss_MIL_fct(prob_sim_matrix_from_v, bs, n_video, n_text)

            prob_sim_matrix_from_t = torch.einsum('ad,bd->ab', [prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim)])     # 与.t()等价
            MIL_loss_t = self.loss_MIL_fct(prob_sim_matrix_from_t, bs, n_video, n_text)
            MIL_loss = (MIL_loss_v + MIL_loss_t) / 2

            ####################################################################
            ############### MCofSoftCOntrastiveLoss ##############
            ####################################################################
            # mcsoft_loss, vib_loss = self.loss_mc_con(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

            ####################################################################
            ############### KL Divergence Loss ##############
            ####################################################################
            vib_loss = self.vib_loss(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)

        # loss
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * wti_logits      # sim_matrix
        if self.training:
            return retrieve_logits, 1e-2 * MIL_loss, 1e-4 * vib_loss
        else:   # for inference 
            return retrieve_logits


    def token_wise_interaction(self, text_token, frame_token, attention_mask, video_mask):
        # token-wise interaction
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_token, frame_token])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = attention_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
        v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits


    def weighted_token_wise_intersection(self, text_token, frame_token, attention_mask, video_mask):
        text_weight = self.text_weight_fc(text_token).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(torch.tensor((1 - attention_mask), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(frame_token).squeeze(2) # B x N_v x D -> B x N_v
        video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        # token-wise interaction
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_token, frame_token])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, attention_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = attention_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits


    def probabilistic_video(self, video_pooled, videos):
        output = {}

        out, attn, residual = self.pie_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output['attention'] = attn
        output['residual'] = residual       # B 512    

        uncertain_out = self.uncertain_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + (residual)         
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma       # B 512     可以看作是方差
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)     # B 512     l2 normalization后 均值
        output['mean'] = out   

        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_video_samples)      # B 7 512    从高斯分布中采样N个embedding  

        return output


    def probabilistic_text(self, text_pooled, text_token):
        output = {}

        out, attn, residual = self.pie_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
        output['attention'] = attn
        output['residual'] = residual

        uncertain_out = self.uncertain_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + (residual)   
        logsigma = uncertain_out['logsigma']
        output['logsigma'] = logsigma
        output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)
        output['mean'] = out

        output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_text_samples)

        return output


    def get_similarity_logits(self, sequence_output, text_token, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
        
        if self.training:
            retrieve_logits, mcsoft_loss, vib_loss = self._loose_similarity(sequence_output, text_token, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
            return retrieve_logits, mcsoft_loss, vib_loss
        else:
            retrieve_logits = self._loose_similarity(sequence_output, text_token, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
            return retrieve_logits,  {}