# D:\code\medical_qa_to_mongo\seven_mongo\question_answers_clean.csv
import jieba
import pandas as pd
import os

# 加载用户词典

jieba.initialize()
entity_path = "../data/entity_dict"
entity_dict = os.listdir(entity_path)
for entity_dict in entity_dict:
    words = pd.read_csv(os.path.join(entity_path, entity_dict), error_bad_lines=False).values.tolist()
    for word in words:
        jieba.add_word(word[0], 999, entity_dict)
answers_clean = pd.read_csv("answers_clean.csv").values.tolist()
with open("answer_train.csv", "w", encoding="utf-8") as answer_train_file:
    for answer_clean in answers_clean:
        if isinstance(answer_clean[0], str):
            answers_clean_words = jieba.lcut(answer_clean[0])
            for answers_clean_words in answers_clean_words:
                answer_train_file.write(answers_clean_words+" ")
            answer_train_file.write("\n")