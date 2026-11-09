import torch
import torch.nn as nn
import torch.nn.functional as F

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.norm = nn.InstanceNorm1d(self.embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = int(model_params['embedding_dim'])
        self.ff_hidden_dim = int(model_params['ff_hidden_dim'])

        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

class MixedScoreAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.head_num = int(model_params['head_num'])
        self.ms_hidden_dim = int(model_params['ms_hidden_dim'])
        self.mix1_init = model_params['ms_layer1_init']
        self.mix2_init = model_params['ms_layer2_init']
        self.qkv_dim = int(self.model_params['qkv_dim'])
        self.sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']
        mix1_weight = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample(sample_shape=(self.head_num, 3, self.ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-self.mix1_init, high=self.mix1_init).sample(sample_shape=(self.head_num, self.ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)

        mix2_weight = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample(sample_shape=(self.head_num, self.ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-self.mix2_init, high=self.mix2_init).sample(sample_shape=(self.head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score, cost_mat_score.transpose(2, 3)), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 3)

        two_scores_transposed = two_scores.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, col_cnt, 3)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1, 2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        if self.model_params['use_sinkhorn']:
            weights = sinkhorn_normalization(mixed_scores)
        else:
            weights = nn.Softmax(dim=3)(mixed_scores)

        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
    return picked_nodes

def reshape_by_heads(qkv, head_num):
    # q shape: (batch, n, head * qkv)
    batch_size = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_size, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head, n, qkv)
    return q_transposed

def sinkhorn_normalization(score, iter=10):
    score = score - score.max(dim=-1, keepdim=True)[0]
    for _ in range(iter):
        score = score - score.logsumexp(dim=-1, keepdim=True)
        score = score - score.logsumexp(dim=-2, keepdim=True)
    return score.exp()

