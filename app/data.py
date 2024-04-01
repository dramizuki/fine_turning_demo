import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_sentences_from_text(filename):
  sentences = []
  with open(filename, 'r') as f:
    for i, line in enumerate(f):
      sentence = line.split('\t')[1].strip()
      if sentence == '': # 空文字を除去
        continue
      if re.match('^http.*$', sentence): # URLを除去
        continue
      sentences.append(sentence)
  return sentences


root_dir = 'data/KNBC_v1.0_090925_utf8/corpus2'
targets = ['Gourmet', 'Keitai', 'Kyoto', 'Sports']

original_data = []
for target in targets:
  filename = os.path.join(root_dir, f'{target}.tsv')
  sentences = get_sentences_from_text(filename)
  for sentence in sentences:
    original_data.append([target, sentence])

original_df = pd.DataFrame(original_data, columns=['target', 'sentence'])

print(original_df.head())
print(original_df.tail())
print(pd.DataFrame(original_df['target'].value_counts()))

train_df, test_df = train_test_split(original_df, test_size=0.1)

train_df.to_csv("./csv/knbc-train.csv")
test_df.to_csv("./csv/knbc-test.csv")