# medical-Wo-BERT
以词为基本单位的中文BERT（Word-based BERT）加入医学专业词典进行训练

## 详情

https://kexue.fm/archives/7758

## 训练

目前开源的WoBERT是Base版本，在哈工大开源的[RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)基础上进行继续预训练，预训练任务为MLM。初始化阶段，将每个词用BERT自带的Tokenizer切分为字，然后用字embedding的平均作为词embedding的初始化。模型使用单张24G的RTX训练了100万步（大概训练了10天），序列长度为512，学习率为5e-6，batch_size为16，累积梯度16步，相当于batch_size=256训练了6万步左右。训练语料大概是30多G的通用型语料。

此外，我们还提供了WoNEZHA，这是基于华为开源的[NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)进行再预训练的，训练细节跟WoBERT基本一样。NEZHA的模型结构跟BERT相似，不同的是它使用了相对位置编码，而BERT用的是绝对位置编码，因此理论上NEZHA能处理的文本长度是无上限的。这里提供以词为单位的WoNEZHA，就是让大家多一个选择。

## 依赖
```cmd
pip install bert4keras==0.8.8
```

## 下载

- **WoBERT**: [chinese_wobert_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1BrdFSx9_n1q2uWBiQrpalw), 提取码: kim2
- **WoNEZHA**: [chinese_wonezha_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1ABKwUuIiMEEsRXxxlbyKmw), 提取码: qgkq

## 引用

Bibtex：

```tex
@techreport{zhuiyipretrainedmodels,
  title={WoBERT: Word-based Chinese BERT model - ZhuiyiAI},
  author={Jianlin Su},
  year={2020},
  url="https://github.com/ZhuiyiTechnology/WoBERT",
}
```

## 联系

邮箱：ai@wezhuiyi.com
追一科技：https://zhuiyi.ai
