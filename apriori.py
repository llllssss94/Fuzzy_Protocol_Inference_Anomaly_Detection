import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def load_data(url):
    raw = pd.read_csv(url)
    raw = raw["Info"].to_list()
    d = []

    for i in raw:
        if type(i) != str:
            i = str(i)
        t = i.split(" ")
        d.append(t)

    return d, raw


def extract(min_sup=0.4, i_url="./data/chat_server.csv", min_num=1):  # 최소 지지도, 인풋 파일 경로, 최소 조합 수

    data, raw = load_data(i_url)

    te = TransactionEncoder()

    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 자주 나오는 키워드 추출
    frequent_items = apriori(df, min_support=min_sup, use_colnames=True)
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))

    wunch_input = []
    # print("APRIORI", frequent_items[frequent_items["length"] > min_num]['itemsets'])
    for line in frequent_items[frequent_items["length"] == 1]['itemsets']:
        if len(line) < 1:
            pass
        temp = ""
        for word in line:
            temp = temp + " " + word
        wunch_input.append(temp[1:])

    return wunch_input

"""
# 키워드간 관련도
# from mlxtend.frequent_patterns import association_rules
# print(association_rules(frequent_items, metric="confidence", min_threshold=0.3))

# 키워드가 포함된 라인 찾기
key = []
for keyword in frequent_items[frequent_items["length"] == 1]["itemsets"]:
    keyword = [x for x in keyword][0]
    if len(keyword) <= 0:
        continue
    key.append(keyword)

meta = []
# find index for each line where keywords are placed
for line in data:
    idx = {"word": [], "idx": []}
    for word in key:
        if word in line:
            idx["word"].append(word)
            idx["idx"].append(line.index(word))
    meta.append(idx)

# pairing raw packet data with keywords
meta = pd.DataFrame(np.array([raw, meta]).T, columns=["raw", "keywords"])


for line in meta.values:
    raw = line[0]
    meta = line[1]
    print(line)
    for i in range(0, len(meta["word"])):
        try:
            idx = meta["idx"].index(i)
            word = meta["word"][idx]
        except:
            continue
"""
if __name__ == "__main__":
    result = extract()
    print(result)

    exit(0)
