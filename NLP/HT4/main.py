import os
import pandas as pd

def isLetter(ch):
    return (ord('a') <= ord(ch) and ord(ch) <= ord('z'))

def isOK(ch):
    return isLetter(ch) or ch.isdigit()

def beatify(str):
    str = str.strip()
    str = str.lower()
    prevSpace = False
    ampRegime = False
    dogRegime = False
    WordDeleted = False
    res = ''
    for ch in str:
        if(not (ampRegime or dogRegime)):
            if (ch == '&'):
                ampRegime = True
            elif (ch == '@'):
                dogRegime = True
                WordDeleted = False
            elif(ch.isspace()):
                if(not prevSpace):
                    res += ' '
                    prevSpace = True
            elif(not isOK(ch)):
                if(not prevSpace):
                    res += ' '
                    prevSpace = True
            else:
                res += ch
                prevSpace = False
        elif(ampRegime):
            if(ch == ';'):
                ampRegime = False
        else:
            if isOK(ch):
                WordDeleted = True
            elif (ch.isspace()):
                if(WordDeleted):
                    dogRegime = False

    return res.strip()

DICTIONARY = {}
d_size = 0
used = []

def getNumber(str):
    global d_size
    id = DICTIONARY.get(str, -1)
    if(id == -1):
        DICTIONARY.update({str: d_size})
        d_size = d_size + 1
        used.append([0, 0])
        return (d_size - 1)
    else:
        return id

def probs(word):
    id = DICTIONARY.get(word, -1)
    if(id == -1):
        return [0.5, 0.5]
    return [float(used[id][0] / (used[id][0] + used[id][1])), float(used[id][1] / (used[id][0] + used[id][1]))]


df_train = pd.read_csv("train.csv", error_bad_lines=False, sep=',', encoding="ISO-8859-1")
df_test = pd.read_csv("test.csv", error_bad_lines=False, sep=',', encoding="ISO-8859-1")

reviews = df_train["SentimentText"].tolist()
result = df_train["Sentiment"].tolist()


total = [0, 0]
for revs, y in zip(reviews, result):
    total[y] = total[y] + 1
    words = beatify(revs).split(" ")
    for word in words:
        id = getNumber(word)
        used[id][y] = used[id][y] + 1

P0 = float(total[0] / (total[0] + total[1]))
P1 = float(total[1] / (total[0] + total[1]))

test_reviews = df_test["SentimentText"].tolist()
res = []
for revs in test_reviews:
    ans = [P0, P1]
    words = beatify(revs).split(" ")
    for word in words:
        p = probs(word)
        ans[0] = ans[0] + p[0]
        ans[1] = ans[1] + p[1]
    if(ans[0] > ans[1]):
        res.append(0)
    else:
        res.append(1)

result = pd.DataFrame()
result["id"] = df_test["ItemId"]
result["sentiment"] = res

result.to_csv("submission.csv", sep=",", index=None, columns=["id", "sentiment"])