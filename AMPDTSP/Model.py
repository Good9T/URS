import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_Lib import (
    AddAndInstanceNormalization,
    FeedForward, MixedScoreAttention,
    _get_encoding,
    reshape_by_heads,
)

class Model(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = Encoder(**self.model_params)
        self.decoder = Decoder(**self.model_params)
        self.embedding_dim = int(self.model_params['embedding_dim'])

        self.node_demand = None
        self.start_node_idx = None
        self.encoded_node = None
        self.k = 10

        self.Projection = nn.Linear(2 * self.k + 1, self.embedding_dim)
        # shape: (batch, node, embedding)

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        dist = problems[..., 0]
        demand = reset_state.node_demand

        mean_val = dist.mean(dim=(1, 2), keepdim=True)
        std_val = dist.std(dim=(1, 2), keepdim=True)
        dist_norm = (dist - mean_val) / (std_val + 1e-9)

        U, S, V = torch.svd_lowrank(dist_norm, q=self.k)
        sqrt_S = torch.sqrt(S)
        Q = U * sqrt_S.unsqueeze(1)
        K = V * sqrt_S.unsqueeze(1)
        X = torch.cat([Q, K], dim=-1)  # (batch, n, 2k)

        total_demand = demand.sum(dim=1, keepdim=True)
        full_demand = demand / (total_demand + 1e-6)
        demand_feat = full_demand.unsqueeze(-1)  # (batch, n, 1)

        X = torch.cat([X, demand_feat], dim=-1)  # (batch, n, 2k+1)

        final_embedding = self.Projection(X)
        self.encoded_node = self.encoder(final_embedding, dist_norm)
        self.decoder.set_kv(self.encoded_node)
        self.start_node_idx = reset_state.start_node_idx

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:
            selected = self.start_node_idx
            prob = torch.ones_like(selected)
            encoded_first_node = _get_encoding(self.encoded_node, selected)
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_current_node = _get_encoding(self.encoded_node, state.current_node)
            all_job_probs = self.decoder(encoded_current_node, mask=state.state_mask)
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
        encoder_layer_num = model_params["encoder_layer_num"]
        self.layers = nn.ModuleList([EncodingBlock(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, node, problem):
        for layer in self.layers:
            node = layer(node, problem)
        return node

class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = int(model_params["embedding_dim"])
        self.head_num = int(model_params["head_num"])
        self.qkv_dim = int(model_params["qkv_dim"])

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.mixed_score_attention = MixedScoreAttention(**model_params)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.normalization1 = AddAndInstanceNormalization(**model_params)
        self.feedforward = FeedForward(**model_params)
        self.normalization2 = AddAndInstanceNormalization(**model_params)

    def forward(self, node, problem):
        q = reshape_by_heads(self.Wq(node), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(node), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(node), head_num=self.head_num)
        # shape: (batch, head, n, key)
        out_concat = self.mixed_score_attention(q, k, v, problem)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.normalization1(node, multi_head_out)
        out2 = self.feedforward(out1)
        out = self.normalization2(out1, out2)

        return out

class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = int(model_params["embedding_dim"])
        self.head_num = int(model_params["head_num"])
        self.qkv_dim = int(model_params["qkv_dim"])
        self.sqrt_qkv_dim = self.model_params["sqrt_qkv_dim"]
        self.sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        self.logit_clipping = self.model_params["logit_clipping"]
        self.Wq0 = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wq1 = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None

    def set_kv(self, encoded_col):
        self.k = reshape_by_heads(self.Wk(encoded_col), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_col), head_num=self.head_num)
        # shape: (batch, head_num, node, )
        self.single_head_key = encoded_col.transpose(1, 2)
        # shape: (batch, embedding, node)

    def set_q1(self, first_row):
        self.q1 = reshape_by_heads(self.Wq1(first_row), head_num=self.head_num)

    def _decode(self, q, k, v, mask=None):
        batch_size = q.size(0)
        n = q.size(2)
        node_num = k.size(2)

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, pomo, node)
        score = score / self.sqrt_qkv_dim
        if mask is not None:
            score = score + mask[:, None, :, :].expand(batch_size, self.head_num, n, node_num)
        weights = nn.Softmax(dim=3)(score)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, n, self.head_num * self.qkv_dim)
        return out

    def forward(self, q0, mask):
        # q0 shape: (batch, pomo, embedding)
        # state_mask shape: (batch, pomo, node)
        q0 = reshape_by_heads(self.Wq0(q0), head_num=self.head_num)
        # shape: (batch, head, pomo, qkv)
        q = self.q1 + q0
        out = self._decode(q, self.k, self.v, mask=mask)
        score = self.multi_head_combine(out)
        score = torch.matmul(score, self.single_head_key)

        score = score / self.sqrt_embedding_dim
        score = self.logit_clipping * torch.tanh(score)
        score = score + mask
        probs = F.softmax(score, dim=2)
        return probs


