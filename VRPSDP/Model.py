import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Model_Lib import FeedForward, reshape_by_heads, multi_head_attention

def _get_encoding(encoded_node, node_index_to_pick):
    # encoded_node.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_node.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_node.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_node = None
        self.node_demand = None
        self.start_node_idx = None

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        self.node_demand = reset_state.node_demand
        self.start_node_idx = reset_state.start_node_idx
        batch_size = problems.size(0)
        node_size = problems.size(1)
        angle = torch.rand(batch_size) * 2 * torch.pi
        cos, sin = torch.cos(angle), torch.sin(angle)

        rot_matrix = torch.stack([
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin, cos], dim=1),
        ], dim=1)
        center = torch.tensor([0.5, 0.5])[None, None, :]
        coords = problems[..., :2]
        others = problems[..., 2:]

        coords_rot = torch.bmm(coords - center, rot_matrix) + center
        problems = torch.cat([coords_rot, others], dim=-1)

        self.encoded_node = self.encoder(problems)
        self.decoder.set_kv(self.encoded_node)

    def forward(self, state, cur_dist, eval_type='greedy'):

        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.POMO_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.zeros(batch_size, pomo_size, dtype=torch.long)
            prob = torch.ones(batch_size, pomo_size)

        elif state.selected_count == 1:
            selected = self.start_node_idx
            prob = torch.ones_like(selected)

        else:

            encoded_last_node = _get_encoding(self.encoded_node, state.current_node)
            # shape: (batch, pomo+1, embedding)

            all_job_probs = self.decoder(encoded_last_node, state.load, cur_dist, ninf_mask=state.state_mask)
            # shape: (batch, pomo+1, problem+1)

            if self.training:
                while True:
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(
                            dim=1).reshape(batch_size, pomo_size)

                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2)
                prob = None

        return selected, prob


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.layer_num = int(model_params['encoder_layer_num'])

        self.emb = nn.Linear(4, self.embedding_dim)
        self.encoded_node = None
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.layer_num)])

    def forward(self, problems):
        emb_problems = self.emb(problems)

        for layer in self.layers:
            emb_problems = layer(emb_problems)

        return emb_problems

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.head_num = int(model_params['head_num'])
        self.qkv_dim = int(model_params['qkv_dim'])

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)

        q = reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=self.head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = input1 + multi_head_out
        out2 = self.feed_forward(out1)
        out3 = out1 + out2

        return out3
        # shape: (batch, problem, embedding)

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.head_num = int(model_params['head_num'])
        self.qkv_dim = int(model_params['qkv_dim'])
        self.logit_clipping = model_params['logit_clipping']

        self.Wq_last = nn.Linear(self.embedding_dim + 1, self.head_num * self.qkv_dim, bias=False)

        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        # IDT & FF
        self.capacity_mapping = nn.Linear(1, self.embedding_dim, bias=False)
        self.feed_forward = FeedForward(**model_params)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_node):
        # encoded_node.shape: (batch, problem+1, embedding)

        self.k = reshape_by_heads(self.Wk(encoded_node), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_node), head_num=self.head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)

        self.single_head_key = encoded_node.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, cur_dist, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        problem_size = ninf_mask.shape[2]
        batch_size = ninf_mask.shape[0]
        pomo_size = ninf_mask.shape[1]

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        q = reshape_by_heads(self.Wq_last(input_cat), head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # IDT
        mh_atten_out = mh_atten_out + encoded_last_node + self.capacity_mapping(load[:, :, None].clone())
        # FF
        q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

        #  Single-Head Attention, for probability calculation
        score = torch.matmul(q_refined, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.embedding_dim ** 0.5


        score_scaled = score / sqrt_embedding_dim

        # -log(dist) heuristic
        score = score_scaled - torch.log(cur_dist)

        score = self.logit_clipping * torch.tanh(score)

        score_masked = score + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs