import numpy as np
import pandas as pd
# 직접 구현
# Global alignment, no gap penalty

def swunch(s1, s2):
    x = len(s1) + 1  # keyword
    y = len(s2) + 1  # target

    mtx = np.zeros((x, y))  # similarity score matrix
    match = 1
    mismatch = 1
    gap = 1  # gap penalty

    mtx[0] = -np.linspace(0, y-1, y)
    mtx = np.transpose(mtx)
    mtx[0] = -np.linspace(0, x-1, x)
    mtx = np.transpose(mtx)

    for xi in range(0, x - 1):
        k = s1[xi]
        for yi in range(0, y - 1):
            if k == s2[yi]:
                score = max(mtx[xi][yi] + match, mtx[xi][yi + 1] - gap, mtx[xi + 1][yi] - gap)
            else:
                score = max(mtx[xi][yi] - mismatch, mtx[xi][yi + 1] - gap, mtx[xi + 1][yi] - gap)
            mtx[xi + 1][yi + 1] = score

    # print(pd.DataFrame(mtx))

    # choose best sequence via total score
    seq1 = ""
    seq2 = ""
    tx, ty = x-1, y-1

    while True:
        t = np.argmax([mtx[tx - 1][ty - 1], mtx[tx - 1][ty], mtx[tx][ty - 1]]) # diagonal first, vertical, horizontal
        # print(tx, ty, " -- ", s2[ty - 1], " | ", t)

        if t == 0:  # diagonal
            seq1 = s1[tx - 1] + seq1
            seq2 = s2[ty - 1] + seq2
            tx = tx - 1
            ty = ty - 1
        elif t == 1:    # up
            seq1 = s1[tx - 1] + seq1
            seq2 = "|" + seq2
            tx = tx - 1
        else:
            seq1 = "|" + seq1
            seq2 = s2[ty - 1] + seq2
            ty = ty - 1

        if tx == 0:
            for i in range(0, ty):
                seq2 = "|" + seq2
            break
        elif ty == 0:
            for i in range(0, tx):
                seq1 = "|" + seq1
            break
    print(seq1, seq2)
    return seq1, seq2


if __name__ == "__main__":
    #swunch("ABCDEEEF", "ABCDEFFF")
    keys = "GET / HTTP/1.0"
    kl = keys.split(" ")
    print(swunch(keys, "GET /index.html HTTP/1.0"))
    """
    keys = "GET / HTTP/1.0"
    kl = keys.split(" ")
    print(swunch(keys, "GET /index.html HTTP/1.0"))
    s1, s2 = swunch(keys, "GET /webtoon/op/ff455fe7f3cf441c9b5ea13500c7e3d09b6240c8 HTTP/1.1")
    print(s1)

    rule = []
    i = 0
    for k in kl:
        if len(rule) > 0:
            rule.append({"len": s1.find(k) - rule[i - 1]["len"], "type": i})
            rule.append({"len": s2.find(k) + len(k) + 1, "type": k})
        else:
            rule.append({"len": s1.find(k) + len(k) + 1, "type": k})
        i += 1

    print(rule)
    """
    exit(0)

