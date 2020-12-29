import pandas as pd
import re
from opencc import OpenCC
from collections import Counter
# from ckiptagger import data_utils, WS, POS
from ckip import CkipSegmenter
segmenter = CkipSegmenter()
import zipfile

# open zipped CKIP model 
# with zipfile.ZipFile("data.zip", 'r') as zip_ref:
#     zip_ref.extractall()

forum = pd.read_csv("forums.csv",names = ["board_name","alias","board_url"])
post_id = pd.read_csv("post_id.csv", names = ["post_id","board_name","post_title","post_excerpt","created_at", "updated_at" ])
post = pd.read_csv("post_content.csv", names = ["post_id","post_title","post_content", "gender", "like_count","comment_count","created_at", "updated_at"])
comment = pd.read_csv("post_comment_new.csv", names = ["comment_id", "post_id","floor", "comment_content", "gender","like_count","created_at", "updated_at"])

def get_alias_by_name(b_name):
  board_name = b_name
  alias = forum.loc[forum.board_name == board_name].alias.values[0]
  return alias 

def get_alias_by_id(p_id):
  p_id = p_id
  alias = post_id.loc[post_id.post_id == p_id].alias.values[0]
  return alias

# clean symbols and spaces 
def cleaning(string):
  if type(string) == str:
    clean_txt = "".join(re.findall(r"[\u4E00-\u9FFF]",string))
  else: 
    clean_txt = ""
  return clean_txt

# tokenization 
# ws = WS("./data")
def tokenization(post):
  try:
    if len(post) > 1:
      result = segmenter.seg(post)
      return result.tok
    else:
      return post
  except:
      return ""
   
# stopwords 
with open("stopwords.txt", encoding="utf-8") as fin:
  stopwords = fin.read().split("\n")[1:]

def no_stop(item):
  no_stop = [x for x in item if x not in stopwords]
  return no_stop

# keywords (for docs more than 100 words)
def keyword(doc):
  keywords = []
  if len(doc) > 100:
    word_count = Counter(doc)
    for w, c in word_count.most_common(3):
      keywords.append(w)
  return keywords

# sentiment 
with open("pos.txt", encoding="utf-8") as pos:
  pos_words = pos.read().split("\n")[1:]

with open("neg.txt", encoding="utf-8") as neg:
  neg_words = neg.read().split("\n")[1:]

def sentiment(token):
  pos = 0
  neg = 0
  for i in token:
    if i in pos_words:
      pos += 1
    elif  i in neg_words:
      neg += 1
  if pos == 0 and neg == 0:
    return "neutral"
  elif pos > neg:
    return "positive"
  else:
    return "negative"

post_id["alias"] = post_id.board_name.apply(get_alias_by_name)
post["alias"] = post.post_id.apply(get_alias_by_id)
comment["alias"] = comment.post_id.apply(get_alias_by_id)
post["url"] = post.alias
comment["url"] = comment.post_id

for i in range(len(post)):
  post["url"][i] = "https://www.dcard.tw/f/" + str(post["alias"][i]) + "/p/" + str(post["post_id"][i])
for i in range(len(comment)):
  comment["url"][i] = "https://www.dcard.tw/f/" + str(comment["alias"][i]) + "/p/" + str(comment["post_id"][i])

post["source"] = "dcard"
comment["source"] = "dcard"
post["type"] = "post"
comment["type"] = "comment"
post["clean_txt"] = post.post_content.apply(cleaning)
comment["clean_txt"] = comment.comment_content.apply(cleaning)

# segmenter.batch_seg(corpus)

# post["token"] = ws(post["clean_txt"])
# comment["token"] = ws(comment["clean_txt"])
post["token"] = post.clean_txt.apply(tokenization)
comment["token"] = comment.clean_txt.apply(tokenization)

post["no_stop"] = post.token.apply(no_stop)
comment["no_stop"] = comment.token.apply(no_stop)
post["keywords"] = post.no_stop.apply(keyword)
comment["keywords"] = comment.no_stop.apply(keyword)
post["sentiment"] = post.token.apply(sentiment)
comment["sentiment"] = comment.token.apply(sentiment)

# save as csv
post.to_csv("clean_post.csv")
comment.to_csv("clean_comment.csv")

