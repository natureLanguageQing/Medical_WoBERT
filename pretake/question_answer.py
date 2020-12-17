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
pd.read_csv("D:\code\medical_qa_to_mongo\seven_mongo\question_answers_clean.csv").answer.drop_duplicates().to_csv(
    "answers_clean.csv", index=False)
