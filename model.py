import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from nlgeval import compute_metrics
from My_NN import EncoderLayer, PositionalEncodingLayer
from repair import get_repair
from utils import read_examples, convert_examples_to_features, read_test_examples

import torch.nn as nn
import torch


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. transformer
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = PositionalEncodingLayer(d_model=config.d_model, max_len=max_length)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        self.dropout = nn.Dropout(p=config.embedding_prob)
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.token_embedding)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None):
        word_embedding = self.token_embedding(source_ids)
        position_embedding = self.position_encoding(source_ids)
        input_embedding = self.dropout(word_embedding) + position_embedding
        # input_embedding = input_embedding.permute([1, 0, 2]).contiguous()
        outputs = self.encoder(input_embedding, src_key_padding_mask = source_mask)
        encoder_output = outputs
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            word_embedding = self.token_embedding(target_ids)
            position_embedding = self.position_encoding(target_ids)
            tgt_embeddings = self.dropout(word_embedding) + position_embedding
            tgt_embeddings = tgt_embeddings.permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)

            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    word_embedding = self.token_embedding(input_ids)
                    position_embedding = self.position_encoding(input_ids)
                    tgt_embeddings = word_embedding + position_embedding
                    tgt_embeddings = tgt_embeddings.permute([1, 0, 2]).contiguous()

                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class DualSC():
    def __init__(self, config_path, beam_size, max_source_length, max_target_length, load_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, tokenizer_class = RobertaConfig, RobertaTokenizer
        config = config_class.from_pretrained(config_path)
        self.tokenizer = tokenizer_class.from_pretrained(config_path)
        # length config
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        self.beam_size = beam_size
        # build model
        encoder = EncoderLayer(d_model= config.d_model, n_heads=config.num_attention_heads, hidden_size=config.hidden_size,
                              dropout=config.hidden_dropout_prob, n_layers=config.num_hidden_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)
        self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=beam_size, max_length=max_target_length,
                        sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id)
        if load_model_path is not None:
            print("从...{}...重新加载参数".format(load_model_path))
            self.model.load_state_dict(torch.load(load_model_path))
        self.model.to(self.device)

    def train(self, train_filename, train_batch_size, num_train_epochs, learning_rate,
              do_eval, dev_filename, eval_batch_size, output_dir, gradient_accumulation_steps=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length, self.max_target_length, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size // gradient_accumulation_steps)

        num_train_optimization_steps = -1

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)
        print("***** 开始训练 *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", train_batch_size)
        print("  Num epoch = %d", num_train_epochs)
        self.model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        for epoch in range(num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                   target_mask=target_mask)

                tr_loss += loss.item()
                train_loss = round(tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            if do_eval==True:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length, self.max_target_length, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("\n***** Running evaluation *****")
                print("  epoch = %d", epoch)
                print("  Num examples = %d", len(eval_examples))
                print("  Batch size = %d", eval_batch_size)

                # Start Evaling model
                self.model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = self.model(source_ids=source_ids, source_mask=source_mask,
                                             target_ids=target_ids, target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                self.model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    print("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    print("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir_ppl = os.path.join(output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir_ppl):
                        os.makedirs(output_dir_ppl)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                eval_examples = read_examples(dev_filename)
                eval_examples = random.sample(eval_examples, min(200, len(eval_examples)))
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='test')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask)
                dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                self.model.eval()
                p = []
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = self.model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                self.model.train()

                csv_pred_list = []
                csv_true_list = []

                for ref, gold in zip(p, eval_examples):
                    csv_true_list.append(gold.target)
                    csv_pred_list.append(ref)

                df = pd.DataFrame(csv_pred_list)
                df.to_csv(os.path.join(output_dir, "valid_hyp.csv"), index=False, header=None)

                df = pd.DataFrame(csv_true_list)
                df.to_csv(os.path.join(output_dir, "valid_ref.csv"), index=False, header=None)

                metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "valid_hyp.csv"),
				                               references=[os.path.join(output_dir, "valid_ref.csv")], no_skipthoughts=True,
				                               no_glove=True)

                dev_bleu = round(metrics_dict['Bleu_4'], 4)
                print("  %s = %s " % ("bleu", str(dev_bleu)))
                print("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    print("  Best bleu:%s", dev_bleu)
                    print("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir_bleu):
                        os.makedirs(output_dir_bleu)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    def test(self, test_filename, test_batch_size, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = []
        files.append(test_filename)
        for idx, file in enumerate(files):
            print("Test file: {}".format(file))
            sum_examples, gen_examples = read_test_examples(file)

            sum_features = convert_examples_to_features(sum_examples, self.tokenizer, self.max_source_length, self.max_target_length, stage='test')
            sum_source_ids = torch.tensor([f.source_ids for f in sum_features], dtype=torch.long)
            sum_source_mask = torch.tensor([f.source_mask for f in sum_features], dtype=torch.long)
            sum_data = TensorDataset(sum_source_ids, sum_source_mask)
            # Calculate bleu
            sum_sampler = SequentialSampler(sum_data)
            sum_dataloader = DataLoader(sum_data, sampler=sum_sampler, batch_size=test_batch_size)
            self.model.eval()
            p = []
            for batch in tqdm(sum_dataloader, total=len(sum_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            sum_pred_list = []
            sum_true_list = []
            for ref, gold in zip(p, sum_examples):
                sum_true_list.append(gold.target)
                sum_pred_list.append(ref)
            df = pd.DataFrame(sum_pred_list)
            df.to_csv(os.path.join(output_dir, "sum_hyp.csv"), index=False, header=None)
            df = pd.DataFrame(sum_true_list)
            df.to_csv(os.path.join(output_dir, "sum_ref.csv"), index=False, header=None)
            metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "sum_hyp.csv"),
				                               references=[os.path.join(output_dir, "sum_ref.csv")], no_skipthoughts=True,
				                               no_glove=True)
            print('ShellCode Summarization:', metrics_dict)

            gen_features = convert_examples_to_features(gen_examples, self.tokenizer, self.max_source_length,
                                                        self.max_target_length, stage='test')
            gen_source_ids = torch.tensor([f.source_ids for f in gen_features], dtype=torch.long)
            gen_source_mask = torch.tensor([f.source_mask for f in gen_features], dtype=torch.long)
            gen_data = TensorDataset(gen_source_ids, gen_source_mask)
            # Calculate bleu
            gen_sampler = SequentialSampler(gen_data)
            gen_dataloader = DataLoader(gen_data, sampler=gen_sampler, batch_size=test_batch_size)
            self.model.eval()
            p = []
            for batch in tqdm(gen_dataloader, total=len(gen_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            gen_pred_list = []
            gen_true_list = []
            count = 0
            for ref, gold in zip(p, gen_examples):
                gen_true_list.append(gold.target)
                data = get_repair(gold.source, ref)
                gen_pred_list.append(data)
                if(data == gold.target):
                    count += 1
            df = pd.DataFrame(gen_true_list)
            df.to_csv(os.path.join(output_dir, "gen_ref.csv"), index=False, header=None)
            df = pd.DataFrame(gen_pred_list)
            df.to_csv(os.path.join(output_dir, "gen_hyp.csv"), index=False, header=None)
            metrics_dict = compute_metrics(hypothesis=os.path.join(output_dir, "gen_hyp.csv"),
                                           references=[os.path.join(output_dir, "gen_ref.csv")], no_skipthoughts=True,
                                           no_glove=True)
            print('ShellCode Generation:', metrics_dict)
            print('ACC:', count/len(gen_true_list))

    def predict(self, source):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.max_source_length, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        source_mask = encode['attention_mask'].to(self.device)
        self.model.eval()
        result_list = []
        with torch.no_grad():
            summary_text_ids = self.model(source_ids=source_ids, source_mask=source_mask)
            for i in range(self.beam_size):
                t = summary_text_ids[0][i].cpu().numpy()
                text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                result_list.append(text)
        return result_list