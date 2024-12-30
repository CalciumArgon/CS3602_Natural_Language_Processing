#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import logging
from text2vec import SentenceModel
# from transformers import BertTokenizer, BertModel, BertConfig
import jieba



class Word2VecJie(nn.Module): # word to vec with jieba
    def __init__(self, device = "cpu"):
        super(Word2VecJie, self).__init__()
        # self.device = device
        self.device = torch.device(f"cuda:{device}" if device>=0 else "cpu")
        print("Word2VecJie device:", device)
        self.text_embed = SentenceModel(device=device)

    # >>>>>>@pengxiang added on 12.15
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
        # model, char_word_pair = Accelerator.prepare(self.text_embed, char_word_pair)
        model = self.text_embed
        # print("char_word_pair:", char_word_pair)
        sentence_length = len(char_word_pair[0])
        char_word_pair = np.array(char_word_pair).reshape(-1) # (B, L) -> (B * L, )

        encoded_words = model.encode(char_word_pair) # (B * L, D)
        # print("encoded_words:", type(encoded_words))
        encoded_words = encoded_words.reshape(-1, sentence_length, model.get_sentence_embedding_dimension()) # (B, L, D)
        
        return torch.Tensor(np.array(encoded_words)).to(self.device)
    
    def forward(self, sentences):
        input_char_word_pair = self._char_word_pair(sentences)
        word_embed = self._word_embed(input_char_word_pair)
        # print(f"device of word_embed:", word_embed.device)

        return word_embed


class SLUTagging_jieba(nn.Module):

    def __init__(self, config):
        super(SLUTagging_jieba, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        # word_embed act as a lookup table, input a word indices, output a word embeddings
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0) # vocab_size: 1741 (words in train.json);  embed_size: 768
        # embed_size: 768; hidden_size: 512; num_layer: 2; num_tags: 74; tag_pad_idx: 0
        self.rnn = getattr(nn, self.cell)(config.embed_size+768, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx) # output prob : (batch_size, seq_len, embed_size)

        self.Jieba_layer = Word2VecJie(device=config.device)
        self.fc_jieba = nn.Linear(768, 768) # 将Word2VecJie的输出过一层线性层

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths # 每个句子真实长度
        sentences = batch.utt

        embed = self.word_embed(input_ids) #  (batch_size, seq_len, embed_size)
        # jieba vector
        jieba_out = self.Jieba_layer(sentences) # (B, L, D)
        jieba_out = self.fc_jieba(jieba_out)
        fused_embeds = torch.cat([embed, jieba_out], dim=-1)  # 拼接，维度变为 (batch_size, seq_len, hidden_size + D)

        packed_inputs = rnn_utils.pack_padded_sequence(fused_embeds, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) # (batch_size, seq_len, hidden_size)

        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        # batch.utt[i] = "导航到凯里大十字" ;from asr_1best
        # pred = [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1]
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-location', 'I-location', 'O', 'B-location']
        # prediction [['inform-对象-导航'], ['inform-序列号-第二个'], ['inform-操作-导航']]

        labels = batch.labels # [[inform-操作-导航, ...], ...] list of list
        output = self.forward(batch)
        prob = output[0] # (batch_size, seq_len, num_tags）
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist() # (seq_len)
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])] # 去掉padding部分
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid) # example: "B-inform-poi名称", "I-inform-poi名称"
                pred_tags.append(tag)
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
            # when history is used, those not in utt_without_history should be removed
            # deal with pred_tuple per item in batch
            if self.config.history:
                pred_tuple_filtered = [p for p in pred_tuple if p.split('-')[-1] in batch.utt_without_history[i]]
                # print("=====================================")
                # print(f"utt_without_history: {batch.utt_without_history[i]}")
                # print(f"utt_with_history: {batch.utt[i]}")
                # print(f"pred_tuple before filter in SLUTagging: {pred_tuple}")
                # print(f"pred_tuple_filtered in SLUTagging: {pred_tuple_filtered}")
                pred_tuple = pred_tuple_filtered

            predictions.append(pred_tuple)


        if len(output) == 1:
            print(f"predictions in SLUTagging: {predictions}") # [['inform-操作-导航'], ['inform-序列号-第二个'], ['inform-操作-导航']] list of list
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
        logits = self.output_layer(hiddens) # (batch_size, seq_len, num_tags）
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32 # 将pad位置的logits设置为负无穷
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
