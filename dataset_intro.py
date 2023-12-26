import pandas as pd
import textwrap

df = pd.read_csv('./datasets/news_summary.csv',encoding = "unicode_escape")

def wrap(x):
  return textwrap.fill(x, width= 100,replace_whitespace = False, fix_sentence_endings = True)

print(df.head(10))
print(df.info()) 
print(f"Read More: \n {wrap(df.loc[2]['read_more'])}")
print(f"Text: \n {wrap(df.loc[2]['text'])}")
print(f"Complete Text \n {wrap(df.loc[2]['ctext'])}")