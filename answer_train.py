#! -*- coding: utf-8 -*-
# 词级别的中文BERT预训练

import json
import os

import jieba
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import multi_gpu_model

from bert4more_health.backend import keras, K
from bert4more_health.layers import Loss
from bert4more_health.models import build_transformer_model
from bert4more_health.optimizers import Adam
from bert4more_health.optimizers import extend_with_gradient_accumulation
from bert4more_health.optimizers import extend_with_weight_decay
from bert4more_health.snippets import DataGenerator
from bert4more_health.snippets import sequence_padding
from bert4more_health.tokenizers import Tokenizer, load_vocab


def text_segment(text, max_length, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > max_length:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > max_length - 1:
                texts.extend(text_segment(text, max_length, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segment(text, max_length, seps[1:], strips))
        return texts
    else:
        return [text]


# 加载用户词典
jieba.initialize()
entity_dict = os.listdir("entity_dict")
for entity_dict in entity_dict:
    words = pd.read_csv(os.path.join("entity_dict", entity_dict), error_bad_lines=False).values.tolist()
    for word in words:
        jieba.add_word(word[0], 999, entity_dict)
# 基本参数
maxlen = 512
batch_size = 16
epochs = 100000
num_words = 20000

# bert配置
config_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/medical_answer_word_bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def corpus(file):
    """语料生成器
    """
    while True:

        with open(file, encoding="utf-8") as f:
            for l in f:
                for text in text_process(l):
                    yield text


def text_process(text):
    """分割文本
    """
    texts = text_segment(text, 32, u'\n。')
    result, length = '', 0
    for text in texts:
        if result and len(result) + len(text) > maxlen * 1.3:
            yield result
            result, length = '', 0
        result += text
    if result:
        yield result


token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
)
pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
user_dict = []
for w, _ in sorted(jieba.dt.FREQ.items(), key=lambda s: -s[1]):
    if w not in token_dict:
        token_dict[w] = len(token_dict)
        user_dict.append(w)
    if len(user_dict) == num_words:
        break
compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
tokenizer_config = open('medical_chinese_word_bert_L-12_H-768_A-12/tokenizer_config.json', 'w', encoding="utf-8")
json.dump([token_dict, keep_tokens, compound_tokens], tokenizer_config, ensure_ascii=False, indent=3)

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            source, target = random_masking(token_ids)
            batch_token_ids.append(source)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                          batch_token_ids, batch_segment_ids, batch_output_ids
                      ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm='linear',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)

AdamW = extend_with_weight_decay(Adam, name='AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, name='AdamWG')
optimizer = AdamWG(
    learning_rate=5e-6,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16,
)
multi_gpu_model(train_model, 2)
train_model.compile(optimizer=optimizer)
train_model.summary()


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """

    def on_epoch_end(self, epoch, logs=None):
        model.save('medical_chinese_word_bert_L-12_H-768_A-12/medical_answer_word_bert_model.h5')  # 保存模型

        model.save_weights('medical_chinese_word_bert_L-12_H-768_A-12/medical_answer_word_bert_model.weights')  # 保存模型


if __name__ == '__main__':

    # 启动训练
    evaluator = Evaluator()
    f = 'data/answers_clean.csv'
    train_generator = data_generator(corpus(f), batch_size, 10 ** 5)

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    # 加载模型 仅仅包含权重
    model.load_weights('medical_chinese_word_bert_L-12_H-768_A-12/medical_answer_word_bert_model.weights')
    # 加载模型 整个模型
    model = load_model("medical_chinese_word_bert_L-12_H-768_A-12/medical_answer_word_bert_model.h5")
