import tqdm
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# threshold_wordcount = np.percentile([len(i) for i in train_data.src] + [len(i) for i in valid_data.src] + [len(i) for i in test_data.src], 97.5)

def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

class ScaleUp(nn.Module):
    """ScaleUp"""

    def __init__(self, scale):
        super(ScaleUp, self).__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale

class PositionalEncodingLayer(nn.Module):

    def __init__(self, d_model, max_len=64):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        positions = torch.arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device)  # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(input_sequences.device)  # [1, d_model]
        angles = self.get_angles(positions, indexes)  # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2])  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2])  # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1)  # [batch_size, seq_len, d_model]
        return position_encoding

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, sequence_threshold=64):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.scaleup = ScaleUp(scaling_factor(sequence_threshold))

    def forward(self, query, key, value, mask):
        """
        :param Tensor[batch_size, q_len, d_model] query
        :param Tensor[batch_size, k_len, d_model] key
        :param Tensor[batch_size, v_len, d_model] value
        :param Tensor[batch_size, ..., k_len] mask
        :return Tensor[batch_size, q_len, d_model] context
        :return Tensor[batch_size, n_heads, q_len, k_len] attention_weights
        """

        Q = self.fc_q(query)  # [batch_size, q_len, d_model]
        K = self.fc_k(key)  # [batch_size, k_len, d_model]
        V = self.fc_v(value)  # [batch_size, v_len, d_model]

        Q = Q.view(Q.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, q_len, head_size]
        K = K.view(K.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, k_len, head_size]
        V = V.view(V.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, v_len, head_size]

        mean = torch.mean(Q, dim=-1)
        mean = mean.unsqueeze(-1)
        Q = Q-mean

        mean = torch.mean(K, dim=-1)
        mean = mean.unsqueeze(-1)
        K = K-mean

        #         scores = torch.matmul(Q, K.transpose(-1, -2)) # [batch_size, n_heads, q_len, k_len]
        #         scores = scores / torch.sqrt(torch.FloatTensor([self.head_size]).to(Q.device))

        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        scaleup = self.scaleup
        scores = scaleup(torch.matmul(Q, K.transpose(-2, -1)))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, q_len, k_len]

        context = torch.matmul(attention_weights, V)  # [batch_size, n_heads, q_len, v_len]
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, q_len, n_heads, v_len]
        context = context.view(context.size(0), -1, self.d_model)
        context = self.fc_o(context)  # [batch_size, q_len, d_model]

        return context, attention_weights

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_model, hidden_size):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(d_model, hidden_size)
        self.fc_ou = nn.Linear(hidden_size, d_model)

    def forward(self, inputs):
        """
        :param Tensor[batch_size, seq_len, d_model] inputs
        :return Tensor[batch_size, seq_len, d_model] outputs
        """
        outputs = F.relu(self.fc_in(inputs))  # [batch_size, seq_len, hidden_size]
        return self.fc_ou(outputs)  # [batch_size, seq_len, d_model]

class EncoderBlockLayer(nn.Module):

    def __init__(self, d_model, n_heads, hidden_size, dropout):
        super(EncoderBlockLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.multi_head_attention_layer = MultiHeadAttentionLayer(d_model=d_model, n_heads=n_heads)
        self.multi_head_attention_layer_norm = nn.LayerNorm(d_model)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_model=d_model, hidden_size=hidden_size)
        self.position_wise_feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_inputs, src_mask):
        """
        :param Tensor[batch_size, src_len, d_model] src_inputs
        :param Tensor[batch_size,  src_len] src_mask
        :return Tensor[batch_size, src_len, d_model] outputs
        """
        context, _ = self.multi_head_attention_layer(query=src_inputs, key=src_inputs, value=src_inputs, mask=src_mask)
        context = self.multi_head_attention_layer_norm(self.dropout(context) + src_inputs)

        outputs = self.position_wise_feed_forward_layer(context)
        outputs = self.position_wise_feed_forward_layer_norm(self.dropout(outputs) + context)

        return outputs

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, hidden_size, dropout, n_layers):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.encoder_block_layers = nn.ModuleList(
            [EncoderBlockLayer(d_model=d_model, n_heads=n_heads, hidden_size=hidden_size,
                               dropout=dropout) for _ in range(n_layers)])
        self.fc_embedding_hidden = nn.Linear(d_model, hidden_size)
        self.fc_hidden_embedding = nn.Linear(hidden_size, d_model)

    def forward(self, input_embedding, src_key_padding_mask):
        """
        :param Tensor[batch_size, src_len] src_sequences
        :param Tensor[batch_size, src_len] src_mask
        :return Tensor[batch_size, src_len, d_model] outputs
        """
        # token_embedded = self.token_embedding(src_sequences)  # [batch_size, src_len, d_model]
        # position_encoded = self.position_encoding(src_sequences)  # [batch_size, src_len, d_model]
        # outputs = self.dropout(token_embedded) + position_encoded  # [batch_size, src_len, d_model]
        outputs = input_embedding
        embedded = outputs
        for layer in self.encoder_block_layers:
            outputs = layer(src_inputs=outputs, src_mask=src_key_padding_mask)  # [batch_size, src_len, d_model]

        outputs = outputs.permute([1, 0, 2]).contiguous() # [src_len, batch_size, d_model]
        return outputs