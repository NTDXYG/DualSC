import os

import tqdm
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Dataset, Field
from torchtext.data.iterator import BucketIterator

from model.beam_utils import Node, find_best_path, find_path

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

SEED = 546
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

train = True
data_dir = "../data"
train_path = "train.csv"
Task = 'ShellCodeGen'
valid_path = 'test.csv'
test_path = 'test_'+Task+'.csv'
D_MODEL = 256
N_LAYERS = 2
save_path = '../save/DualSC.pth'
N_HEADS = 8
HIDDEN_SIZE = 512
DROPOUT = 0.25
BATCH_SIZE = 32
LR = 1e-3
N_EPOCHS = 30
CODE_MAX_LEN = 50
NL_MAX_LEN = 50
GRAD_CLIP = 1.0


def tokenize_code(text):
    return text.split()

def tokenize_nl(text):
    return text.split()

CODE = Field(tokenize = tokenize_code,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

NL = Field(tokenize = tokenize_nl,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

from torchtext import data
train_data, valid_data, test_data = data.TabularDataset.splits(path=data_dir,
                                              train=train_path,
                                              validation=valid_path,
                                              test=test_path,
                                              format='csv',
                                              skip_header=True,
                                              csv_reader_params={'delimiter':','},
                                              fields=[('src',CODE),('trg',NL)])
MIN_COUNT = 1
CODE.build_vocab(train_data, min_freq=MIN_COUNT)
NL.build_vocab(train_data, min_freq=MIN_COUNT)

print(f'Length of CODE vocabulary: {len(CODE.vocab):,}')
print(f'Length of NL vocabulary: {len(NL.vocab):,}')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : -len(x.src),
     device = DEVICE)

threshold_wordcount = np.percentile([len(i) for i in train_data.src] + [len(i) for i in valid_data.src] + [len(i) for i in test_data.src], 97.5)

def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

class ScaleUp(nn.Module):
    """ScaleUp"""
    def __init__(self, scale):
        super(ScaleUp, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, sequence_threshold=threshold_wordcount):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.scaleup = ScaleUp(scaling_factor(sequence_threshold))
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

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
        # scores = torch.matmul(Q, K.transpose(-1, -2)) # [batch_size, n_heads, q_len, k_len]
        # scores = scores / torch.sqrt(torch.FloatTensor([self.head_size]).to(Q.device))
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        scaleup = self.scaleup
        scores = scaleup(torch.matmul(Q, K.transpose(-2, -1)))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e18)
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

class PositionalEncodingLayer(nn.Module):

    def __init__(self, d_model, max_len=100):
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
        return outputs, _

class EncoderLayer(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, n_heads, hidden_size, dropout, n_layers):
        super(EncoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncodingLayer(d_model=d_model, max_len=max_len)
        self.encoder_block_layers = nn.ModuleList(
            [EncoderBlockLayer(d_model=d_model, n_heads=n_heads, hidden_size=hidden_size,
                               dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_sequences, src_mask):
        """
        :param Tensor[batch_size, src_len] src_sequences
        :param Tensor[batch_size, src_len] src_mask
        :return Tensor[batch_size, src_len, d_model] outputs
        """
        token_embedded = self.token_embedding(src_sequences)  # [batch_size, src_len, d_model]
        position_encoded = self.position_encoding(src_sequences)  # [batch_size, src_len, d_model]
        outputs = self.dropout(token_embedded) + position_encoded  # [batch_size, src_len, d_model]
        for layer in self.encoder_block_layers:
            outputs, enc_attention = layer(src_inputs=outputs, src_mask=src_mask)  # [batch_size, src_len, d_model]
        return outputs, enc_attention

class DecoderBlockLayer(nn.Module):

    def __init__(self, d_model, n_heads, hidden_size, dropout):
        super(DecoderBlockLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.mask_multi_head_attention_layer = MultiHeadAttentionLayer(d_model=d_model, n_heads=n_heads)
        self.mask_multi_head_attention_layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention_layer = MultiHeadAttentionLayer(d_model=d_model, n_heads=n_heads)
        self.multi_head_attention_layer_norm = nn.LayerNorm(d_model)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_model=d_model, hidden_size=hidden_size)
        self.position_wise_feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, dest_inputs, src_encoded, dest_mask, src_mask):
        """
        :param Tensor[batch_size, dest_len, d_model] dest_inputs
        :param Tensor[batch_size, src_len, d_model] src_encoded
        :param Tensor[batch_size,  dest_len] dest_mask
        :param Tensor[batch_size,  src_len] src_mask
        :return Tensor[batch_size, dest_len, d_model] outputs
        :return Tensor[batch_size, n_heads, dest_len, src_len] attention_weights
        """

        masked_context, _ = self.mask_multi_head_attention_layer(query=dest_inputs, key=dest_inputs, value=dest_inputs,
                                                                 mask=dest_mask)
        masked_context = self.mask_multi_head_attention_layer_norm(self.dropout(masked_context) + dest_inputs)

        context, attention_weights = self.multi_head_attention_layer(query=masked_context, key=src_encoded,
                                                                     value=src_encoded, mask=src_mask)
        context = self.multi_head_attention_layer_norm(self.dropout(context) + masked_context)

        outputs = self.position_wise_feed_forward_layer(context)
        outputs = self.position_wise_feed_forward_layer_norm(self.dropout(outputs) + context)
        return outputs, attention_weights

class DecoderLayer(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, n_heads, hidden_size, dropout, n_layers):
        super(DecoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncodingLayer(d_model=d_model, max_len=max_len)
        self.decoder_block_layers = nn.ModuleList(
            [DecoderBlockLayer(d_model=d_model, n_heads=n_heads, hidden_size=hidden_size, dropout=dropout) for _ in
             range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, dest_sequences, src_encoded, dest_mask, src_mask):
        """
        :param Tensor[batch_size, dest_len] dest_sequences
        :param Tensor[batch_size, src_len, d_model] src_encoded
        :param Tensor[batch_size, dest_len, d_model] dest_mask
        :param Tensor[batch_size, src_len, d_model] src_mask
        :return Tensor[batch_size, dest_len, vocab_size] logits
        :return Tensor[batch_size, n_heads, dest_len, src_len] attention_weights
        """
        token_embedded = self.token_embedding(dest_sequences)  # [batch_size, dest_len, d_model]
        position_encoded = self.position_encoding(dest_sequences)  # [batch_size, dest_len, d_model]
        outputs = self.dropout(token_embedded) + position_encoded  # [batch_size, dest_len, d_model]
        for layer in self.decoder_block_layers:
            outputs, attention_weights = layer(dest_inputs=outputs, src_encoded=src_encoded, dest_mask=dest_mask,
                                               src_mask=src_mask)
        logits = self.fc(outputs)
        return logits, attention_weights

class Transformer(nn.Module):

    def __init__(self, encoder, decoder, src_pad_index, dest_pad_index):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_index = src_pad_index
        self.dest_pad_index = dest_pad_index

    def make_src_mask(self, src_sequences):
        """Mask <pad> tokens.
        :param Tensor[batch_size, src_len] src_sequences
        :return Tensor[batch size, 1, 1, src len] src_mask
        """
        src_mask = (src_sequences != self.src_pad_index).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_dest_mask(self, dest_sequences):
        """Mask <pad> tokens and futur tokens as well.
        :param Tensor[batch_size, dest_len] dest_sequences
        :return tensor[batch_size, 1, dest_len, dest_len] dest_mask
        """
        mask = (dest_sequences != self.dest_pad_index).unsqueeze(1).unsqueeze(2)  # [batch size, 1, 1, trg len]
        sub_mask = torch.tril(torch.ones((dest_sequences.size(1), dest_sequences.size(1))).to(
            dest_sequences.device)).bool()  # [trg len, trg len]
        return mask & sub_mask

    def forward(self, src_sequences, dest_sequences):
        """
        :param Tensor[batch_size, src_len] src_sequences
        :param Tensor[batch_size, dest_len] dest_sequences
        :return Tensor[batch_size, dest_len, vocab_size] logits
        :return Tensor[batch_size, n_heads, dest_len, src_len] attention_weights
        """
        src_mask, dest_mask = self.make_src_mask(src_sequences), self.make_dest_mask(dest_sequences)
        src_encoded, enc_attention = self.encoder(src_sequences=src_sequences, src_mask=src_mask)
        logits, attention_weights = self.decoder(dest_sequences=dest_sequences, src_encoded=src_encoded,
                                                 dest_mask=dest_mask, src_mask=src_mask)
        return logits, attention_weights

class AverageMeter:

    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def accuracy(outputs, target_sequences, k=5):
    """ Calculate Top-k accuracy
    :param Tensor[batch_size, dest_seq_len, vocab_size] outputs
    :param Tensor[batch_size, dest_seq_len] target_sequences
    :return float Top-k accuracy
    """
    # print([*map(lambda token: EN.vocab.itos[token], outputs.argmax(dim=-1)[0].tolist())])
    # print([*map(lambda token: EN.vocab.itos[token], target_sequences[0].tolist())])
    # print("="*100)
    batch_size = target_sequences.size(0)
    _, indices = outputs.topk(k, dim=2, largest=True, sorted=True)  # [batch_size, dest_seq_len, 5]
    correct = indices.eq(target_sequences.unsqueeze(-1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / indices.numel())

class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, loader, epoch, grad_clip):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.train()
        progress_bar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for i, batch in progress_bar:
            src, trg = batch.src, batch.trg
            self.optimizer.zero_grad()
            logits, _ = self.model(src, trg[:, :-1])  # [batch_size, dest_len, vocab_size]
            loss = self.criterion(logits.contiguous().view(-1, self.model.decoder.vocab_size),
                                  trg[:, 1:].contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            loss_tracker.update(loss.item())
            acc_tracker.update(accuracy(logits, trg[:, 1:]))
            loss_, ppl_, acc_ = loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average
            progress_bar.set_description(
                f'Epoch: {epoch + 1:02d} -     loss: {loss_:.3f} -     ppl: {ppl_:.3f} -     acc: {acc_:.3f}%')
        return loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average

    def validate(self, loader, epoch):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(loader), total=len(loader))
            for i, batch in progress_bar:
                src, trg = batch.src, batch.trg
                logits, _ = self.model(src, trg[:, :-1])  # [batch_size, dest_len, vocab_size]
                loss = self.criterion(logits.contiguous().view(-1, self.model.decoder.vocab_size),
                                      trg[:, 1:].contiguous().view(-1))
                loss_tracker.update(loss.item())
                acc_tracker.update(accuracy(logits, trg[:, 1:]))
                loss_, ppl_, acc_ = loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average
                progress_bar.set_description(
                    f'Epoch: {epoch + 1:02d} - val_loss: {loss_:.3f} - val_ppl: {ppl_:.3f} - val_acc: {acc_:.3f}%')
        return loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average

    def train(self, train_loader, valid_loader, n_epochs, grad_clip):
        history, best_loss = {'acc': [], 'loss': [], 'ppl': [], 'val_ppl': [], 'val_acc': [], 'val_loss': []}, np.inf
        for epoch in range(n_epochs):
            loss, ppl, acc = self.train_step(train_loader, epoch, grad_clip)
            val_loss, val_ppl, val_acc = self.validate(valid_loader, epoch)
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
            history['acc'].append(acc);
            history['val_acc'].append(val_acc)
            history['ppl'].append(ppl);
            history['val_ppl'].append(val_ppl)
            history['loss'].append(loss);
            history['val_loss'].append(val_loss)
        return history

transformer = Transformer(
    encoder=EncoderLayer(
        vocab_size=len(CODE.vocab),
        max_len=CODE_MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ),
    decoder=DecoderLayer(
        vocab_size=len(NL.vocab),
        max_len=NL_MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ),
    src_pad_index=CODE.vocab.stoi[CODE.pad_token],
    dest_pad_index=NL.vocab.stoi[NL.pad_token]
).to(DEVICE)

optimizer = optim.Adam(params=transformer.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=NL.vocab.stoi[NL.pad_token])
print(f'Number of parameters of the model: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}')


def translate(sentences, model, beam_size, src_field, dest_field, max_len, device):
    if isinstance(sentences, list):
        sentences = [*map(src_field.preprocess, sentences)]
        targets = None
    if isinstance(sentences, Dataset):
        targets = [*map(lambda example: ' '.join(example.trg), sentences.examples)]
        sentences = [*map(lambda example: example.src, sentences.examples)]
    data = [*map(lambda word_list: src_field.process([word_list]), sentences)]
    sentences = [*map(lambda sentence: ' '.join(sentence), sentences)]
    translated_sentences, attention_weights, pred_logps = [], [], []
    model.eval()
    with torch.no_grad():
        for i, src_sequence in tqdm.tqdm(enumerate(data), total=len(data)):
            src_sequence = src_sequence.to(device)
            src_mask = model.make_src_mask(src_sequence)
            src_encoded, enc_attention = model.encoder(src_sequences=src_sequence, src_mask=src_mask)
            tree = [
                [Node(token=torch.LongTensor([dest_field.vocab.stoi[dest_field.init_token]]).to(device), states=())]]
            for _ in range(max_len):
                next_nodes = []
                for node in tree[-1]:
                    if node.eos:  # Skip eos token
                        continue
                    # Get tokens that're already translated
                    already_translated = torch.LongTensor(
                        [*map(lambda node: node.token, find_path(tree))][::-1]).unsqueeze(0).to(device)
                    dest_mask = model.make_dest_mask(already_translated)
                    logit, attn_weights = model.decoder(dest_sequences=already_translated, src_encoded=src_encoded,
                                                        dest_mask=dest_mask,
                                                        src_mask=src_mask)  # [1, dest_seq_len, vocab_size]
                    logp = F.log_softmax(logit[:, -1, :], dim=1).squeeze(
                        dim=0)  # [vocab_size] Get scores
                    topk_logps, topk_tokens = torch.topk(logp,
                                                         beam_size)  # Get top k tokens & logps
                    for k in range(beam_size):
                        next_nodes.append(Node(token=topk_tokens[k, None], states=(attn_weights,),
                                               logp=topk_logps[k, None].cpu().item(), parent=node,
                                               eos=topk_tokens[k].cpu().item() == dest_field.vocab[
                                                   dest_field.eos_token]))
                if len(next_nodes) == 0:
                    break
                next_nodes = sorted(next_nodes, key=lambda node: node.logps, reverse=True)
                tree.append(next_nodes[:beam_size])
            best_path = find_best_path(tree)[::-1]
            # Get the translation
            pred_translated = [*map(lambda node: dest_field.vocab.itos[node.token], best_path)]
            pred_translated = [*filter(lambda word: word not in [
                dest_field.init_token, dest_field.eos_token
            ], pred_translated)]
            translated_sentences.append(' '.join(pred_translated))
            # Get probabilities
            pred_logps.append(sum([*map(lambda node: node.logps, best_path)]))
            # Get attention weights
            attention_weights.append(best_path[-1].states[0].cpu().numpy())
    return sentences, translated_sentences, targets, attention_weights, pred_logps, enc_attention

if(train):
    if(os.path.exists(save_path)):
        transformer.load_state_dict(torch.load(save_path))
        transformer.to(DEVICE)
    trainer = Trainer(model=transformer, optimizer=optimizer, criterion=criterion)
    history = trainer.train(train_loader=train_iterator, valid_loader=valid_iterator, n_epochs=N_EPOCHS, grad_clip=GRAD_CLIP)
else:
    transformer.load_state_dict(torch.load(save_path))
    transformer.to(DEVICE)

    sentences, translated_sentences, targets, attention_weights, pred_logps, enc_attention = translate(sentences=test_data, model=transformer, beam_size=1, src_field=CODE,
                                                                                               dest_field=NL, max_len=NL_MAX_LEN, device=DEVICE)

    pred_df = pd.DataFrame(translated_sentences)
    pred_df.to_csv('../result/Trans_'+Task+'_adj.csv', index=None, header=False)
