#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import os

root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BertSLU(nn.Module):

    def __init__(self, config):
        super(BertSLU, self).__init__()
        self.config = config

        self.device = config.device if config.device >= 0 else torch.device("cpu")
        self._init_bert_model(config)

        if config.use_bert_state == 'fuse':
            self.fuse_embed_layer = nn.GRU(self.bert_hidden_size * 4, self.bert_hidden_size, bidirectional=False, batch_first=True)
        else:
            self.fuse_embed_layer = None

        self.num_tags = config.num_tags
        print(f"Number of tags: {self.num_tags}")
        assert False
        if config.decoder_cell == "FNN":
            self.output_layer = TaggingFNNDecoder(self.bert_hidden_size, self.num_tags, config.tag_pad_idx)
        else:
            self.output_layer = TaggingRNNDecoder(self.bert_hidden_size, self.num_tags, config.tag_pad_idx, config.decoder_cell)


    def _init_bert_model(self, cfg):

        if (cfg.encoder_cell).lower() == "bert":
            self.bert_cfg = BertConfig.from_pretrained('bert-base-chinese', cache_dir=os.path.join(root_path, 'cache'))
            self.bert_hidden_size = self.bert_cfg.hidden_size
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=os.path.join(root_path, 'cache'))
            self.model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True, cache_dir=os.path.join(root_path, 'cache')).to(self.device)
        elif (cfg.encoder_cell).lower() == "macbert":
            self.bert_hidden_size = 768
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", cache_dir=os.path.join(root_path, 'cache'))
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base", output_hidden_states=True, cache_dir=os.path.join(root_path, 'cache')).to(self.device)
            return
        elif (cfg.encoder_cell).lower() == "robert":
            self.bert_cfg = BertConfig.from_pretrained('roberta-base', cache_dir=os.path.join(root_path, 'cache'))
            self.bert_hidden_size = self.bert_cfg.hidden_size
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base", cache_dir=os.path.join(root_path, 'cache'))
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states=True,
                                                   cache_dir=os.path.join(root_path, 'cache'), ).to(self.device)
        else:
            raise ValueError("[CS3602] Model type {} not supported".format(self.encoder_cell))

        self.bert_cfg.word2vec_embed_size = cfg.embed_size
        self.bert_cfg.word2vec_vocab_size = cfg.vocab_size

        if cfg.lock_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            lock_num = int(len(self.model.encoder.layer) * cfg.lock_bert_ratio)
            for layer in self.model.encoder.layer[:lock_num]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        lengths = batch.lengths

        encoded_inputs = self.tokenizer(batch.utt,
                                padding="max_length",
                                truncation=True,
                                max_length=max(lengths),
                                return_tensors='pt').to(self.device)
        hidden_states = self.model(**encoded_inputs).hidden_states

        if self.config.use_bert_state == 'last':
            embed = hidden_states[-1]
        elif self.config.use_bert_state == 'mean':
            embed = torch.mean(torch.stack(hidden_states[-4:], dim=0), dim=0)
        elif self.config.use_bert_state == 'fuse':
            B, L = hidden_states[-1].shape[:2]
            stacked_hidden_states = torch.stack(hidden_states[-4:]).view(B, L, -1)
            embed, _ = self.fuse_embed_layer(stacked_hidden_states)
        
        tag_output = self.output_layer(embed, tag_mask, tag_ids)
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []

        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                # 'O' serves as a separator, 'B' symbols the start and 'I' symbols mid-word. This is used for POS-Tagging later.
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )


class TaggingRNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id, decoder_model_type="GRU", num_layers=1):
        super(TaggingRNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.feat_dim = 100
        self.output_layer = getattr(nn, decoder_model_type)(input_size, self.feat_dim, num_layers=num_layers, bidirectional=True)
        self.linear_layer = nn.Linear(self.feat_dim * 2, num_tags)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)[0]
        logits = self.linear_layer(logits)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_function(logits.reshape(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
