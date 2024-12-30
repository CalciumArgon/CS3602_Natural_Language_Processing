#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import os
import jieba
from text2vec import SentenceModel
import logging
import numpy as np

root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = root_path

class FuseBertSLU(nn.Module):

    def __init__(self, config):
        super(FuseBertSLU, self).__init__()
        self.config = config

        self.device = config.device if config.device >= 0 else 'cpu'
        self._init_bert_model(config)

        # self.fuse_model = ChunkWordModel(config, self.bert_cfg)
        self.Jieba_layer = Word2VecJie(device=config.device)
        self.fc_jieba = nn.Linear(768, 768)

        if config.use_bert_state == 'fuse':
            self.fuse_embed_layer = nn.GRU(self.bert_hidden_size * 4, self.bert_hidden_size, bidirectional=False, batch_first=True)
        else:
            self.fuse_embed_layer = None

        self.num_tags = config.num_tags
        ###### 在 fuse 的情况下输出 embed 形状是 (B,L,2D)
        if config.decoder_cell == "FNN":
            self.output_layer = TaggingFNNDecoder(2 * self.bert_hidden_size, self.num_tags, config.tag_pad_idx)
        else:
            self.output_layer = TaggingRNNDecoder(2 * self.bert_hidden_size, self.num_tags, config.tag_pad_idx, config.decoder_cell)


    def _init_bert_model(self, cfg):
        print("Finding BERT model in {}".format(os.path.join(root_path, 'cache')))
        if (cfg.encoder_cell).lower() == "bert":
            self.bert_cfg = BertConfig.from_pretrained('bert-base-chinese', cache_dir=os.path.join(root_path, 'cache'))
            self.bert_hidden_size = self.bert_cfg.hidden_size
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=os.path.join(root_path, 'cache'))
            self.model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True, cache_dir=os.path.join(root_path, 'cache'), mirror='tuna').to(self.device)
        elif (cfg.encoder_cell).lower() == "macbert":
            self.bert_hidden_size = 768
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", cache_dir=os.path.join(root_path, 'cache'))
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base", output_hidden_states=True, cache_dir=os.path.join(root_path, 'cache'), mirror='tuna').to(self.device)
            return
        elif (cfg.encoder_cell).lower() == "robert":
            self.bert_cfg = BertConfig.from_pretrained('clue/roberta-base', cache_dir=os.path.join(root_path, 'cache'))
            self.bert_hidden_size = self.bert_cfg.hidden_size
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base", cache_dir=os.path.join(root_path, 'cache'))
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states=True,
                                                   cache_dir=os.path.join(root_path, 'cache'), mirror='tuna').to(self.device)
        else:
            assert False, "[CS3602] Model type {} not supported".format(self.encoder_cell)

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
            
        # embed = self.fuse_model(batch.utt, embed)
        jieba_out = self.Jieba_layer(batch.utt)
        jieba_out = self.fc_jieba(jieba_out)
        embed = torch.cat([embed, jieba_out], dim=-1)
        
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


class Word2VecJie(nn.Module): # word to vec with jieba
    def __init__(self, device):
        super(Word2VecJie, self).__init__()
        self.device = torch.device(f"cuda:{device}" if device>=0 else "cpu")
        print("Word2VecJie device:", device)
        self.text_embed = torch.load(os.path.join(root_path, 'cache/sentencemodel'), map_location=self.device)
        print("Load SentenceModel from {}".format(os.path.join(root_path, 'cache/sentencemodel')))
        # self.text_embed = SentenceModel(device=self.device)
        # torch.save(self.text_embed, os.path.join(root_path, 'cache/sentencemodel'))
        # assert False, "save SentenceModel to {}".format(os.path.join(root_path, 'cache/sentencemodel'))

    def _char_word_pair(self, batch_sentences, pad_length=None):
        """
            transform a batch of Chinese sentence into a list of char-word pairs
            Example: ["中国一定加油", "我要学自然语言处理"] ( -> [['中国', '一定', '加油'], ['我', '要学', '自然语言', '处理']] )
                        -> [['中国', '中国', '一定', '一定', '加油', '加油', '', '', ''], ['我', '要学', '要学', '自然语言', '自然语言', '自然语言', '自然语言', '处理', '处理']]
            
        Args:
            batch_sentences (list): A list of (unpadded) Chinese sentences
        """
        jieba.setLogLevel(logging.ERROR)

        if pad_length is None:
            pad_length = max([len(each) for each in batch_sentences])

        cut_words = []
        for sentence in batch_sentences:
            # result = jieba.tokenize(sentence, mode="default")
            cut_word = list(jieba.cut(sentence, cut_all=False))
            cut_words.append(cut_word)
        # print("cut_words: ", cut_words) # [['中国', '一定', '加油'], ['我', '要学', '自然语言', '处理']]

        batch_pair_list = []
        for cut_word in cut_words:
            pair_list = []
            for word in cut_word: # cut_word: ['中国', '一定', '加油']; word: '中国'
                for i in range(len(word)):
                    pair_list.append(word)
            batch_pair_list.append(pair_list)
        # print("batch_pair_list:", batch_pair_list) # 

        for i in range(len(batch_pair_list)):
            batch_pair_list[i].extend(["" for _ in range(pad_length - len(batch_pair_list[i]))])

        return batch_pair_list

    def _word_embed(self, char_word_pair):
        """Takes in original char_word_pairs and transform them into vectors using text2vec sentence model.

        Args:
            char_word_pair (list [list [list]]): [[["美国"], ["美国"], ["人民"], ["人民"], [""]] ..]
        """
        model = self.text_embed
        # print("char_word_pair:", char_word_pair)
        sentence_length = len(char_word_pair[0])
        char_word_pair = np.array(char_word_pair).reshape(-1) # (B, L) -> (B * L, )

        encoded_words = model.encode(char_word_pair) # (B * L, D)
        encoded_words = encoded_words.reshape(-1, sentence_length, model.get_sentence_embedding_dimension()) # (B, L, D)
        
        return torch.Tensor(np.array(encoded_words)).to(self.device)
    
    def forward(self, sentences):
        input_char_word_pair = self._char_word_pair(sentences)
        word_embed = self._word_embed(input_char_word_pair)
        return word_embed


'''
class ChunkWordModel(nn.Module):
    """
        A fusion model that takes in a charactor vector and the paired word features.
        Adapted from https://github.com/liuwei1206/LEBERT/blob/main/wcbert_modeling.py
    """

    def __init__(self, cfg, bertcfg):
        super(ChunkWordModel, self).__init__()
        self.dropout = nn.Dropout(0.2)
        # self.tanh = nn.Tanh()
        self.device = cfg.device if cfg.device >= 0 else 'cpu'

        # self.text_embed = SentenceModel(device = self.device)
        # self.text_embed = SentenceModel(model_name_or_path=root_path, device=self.device)
        self.text_embed = AutoModel.from_pretrained(os.path.join(root_path, 'cache/tiny-bert')).to(self.device)
        self.text_embed_tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_path, 'cache/tiny-bert'))

        # self.hidden_decode = getattr(nn, bertcfg.fuse_decoder)(cfg.hidden_size,
        #                                                         cfg.hidden_size,
        #                                                         bidirectional=False,
        #                                                         batch_first=True)

        # self.word_transform = nn.Linear(bertConfig.word2vec_embed_size, bertConfig.hidden_size)
        # self.word_word_weight = nn.Linear(bertConfig.hidden_size, bertConfig.hidden_size)
        # attn_W = torch.zeros(bertConfig.hidden_size, bertConfig.hidden_size)
        # self.attn_W = nn.Parameter(attn_W)
        # self.attn_W.data.normal_(mean=0.0, std=bertConfig.initializer_range)

        ########## 这里修改为 2D 大小, 后面要两个 embedding 拼接 ##########
        self.fuse_layernorm = nn.LayerNorm(2 * cfg.hidden_size, eps=bertcfg.layer_norm_eps)

    # >>>>>>@pengxiang added on 12.15
    def _char_word_pair(self, batch_sentences, pad_length=None):
        """
            transform a batch of Chinese sentence into a list of char-word pairs
            Example: "美国人民" -> ["美国", "美国", "人民", "人民"]
            
        Args:
            batch_sentences (list): A list of (unpadded) Chinese sentences
        """
        jieba.setLogLevel(logging.ERROR)

        if pad_length is None:
            pad_length = max([len(each) for each in batch_sentences])

        cut_words = []
        for sentence in batch_sentences:
            # result = jieba.tokenize(sentence, mode="default")
            cut_word = list(jieba.cut(sentence, cut_all=True))
            cut_words.append(cut_word)

        batch_pair_list = []
        for i, sentence_cut_words in enumerate(cut_words):
            pair_list = [set() for _ in range(len(batch_sentences[i]))]  # use set to prevent redundance.
            for each_word in sentence_cut_words:
                for character in each_word:
                    indices = [i for i, targ in enumerate(batch_sentences[i]) if targ == character]

                    for idx in indices:
                        pair_list[idx].add(each_word)

            batch_pair_list.append(pair_list)

        # Transform set back to list and record W.
        W = 0
        for pair_list in batch_pair_list:
            for i in range(len(pair_list)):
                pair_list[i] = list(pair_list[i])
                W = max(len(pair_list[i]), W)

        # Pad the word length of each sentence to pad_length
        for i in range(len(batch_pair_list)):
            batch_pair_list[i].extend([[""] for _ in range(pad_length - len(batch_pair_list[i]))])

        # Pad all the paired lists of each char in each sentence to W
        for pair_list in batch_pair_list:
            for i in range(len(pair_list)):
                pair_list[i].extend(["" for _ in range(W - len(pair_list[i]))])

        return batch_pair_list

    def _word_embed(self, char_word_pair):
        """Takes in original char_word_pairs and transform them into vectors using text2vec sentence model.

        Args:
            char_word_pair (list [list [list]]): [[["美国"], ["美国"], ["人民"], ["人民"], [""]] ..]
        """
        # model = self.text_embed
        words_to_encode = [word for pair_list in char_word_pair for pair in pair_list for word in pair if word]
        # # 批量编码
        # encoded_words = model.encode(words_to_encode)
        # # 处理空字符串的情况
        # default_vector = model.encode("")  # 假设模型有一个方法来获取嵌入维度

        def encode(str_lst):
            inputs = self.text_embed_tokenizer(str_lst, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.text_embed(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
            return outputs

        encoded_words = encode(words_to_encode)
        default_vector = encode("")

        # 将编码结果分配回原始数据结构
        idx = 0
        for pair_list in char_word_pair:
            for pair in pair_list:
                for i in range(len(pair)):
                    if pair[i]:
                        pair[i] = encoded_words[idx]
                        idx += 1
                    else:
                        pair[i] = default_vector

        # 转换为张量
        embedded_char_word_pair = torch.Tensor(np.array(char_word_pair))

        return embedded_char_word_pair

    def process_sentence(self, input_sentence):
        """
            Inputs:
            input_sentence: a list of length B that stores original Chinese sentences in a batch
        """
        # print("input_sentence = ", input_sentence)
        input_char_word_pair = self._char_word_pair(input_sentence)
        # print("input_char_word_pair = ", input_char_word_pair)
        word_embed = self._word_embed(input_char_word_pair)
        # print("(input_)word_embed.shape=", word_embed.shape)
        return word_embed

    def forward(self, input_sentence, layer_output):
        """
            Inputs:
            input_sentence: a list of length B that stores original Chinese sentences in a batch
            layer_output: The result of a hidden layer of size [B, L, D] where L is the padded length and D is the feature dim.
        """
        input_word_embeddings = self.process_sentence(input_sentence)  # [B, ?] -> [B, L, W, D]
        # B, L, W, D = input_word_embeddings.shape
        # print(f"B={B}, L={L}, W={W}, D={D}")
        input_word_embeddings = input_word_embeddings.squeeze()
        input_word_embeddings = input_word_embeddings.to(layer_output.device)
        # layer_output, _ = self.hidden_decode(layer_output)
        # print("layer_output.shape=", layer_output.shape)

        # 接下来把 input_word_embeddings 和 layer_output 并联到一起变成 (B,L,2D)
        assert layer_output.shape == input_word_embeddings.shape, "[tr] two embedding not same shape"
        layer_output = torch.cat((layer_output, input_word_embeddings), dim=-1)

        layer_output = self.dropout(layer_output)
        layer_output = self.fuse_layernorm(layer_output)

        return layer_output
'''
