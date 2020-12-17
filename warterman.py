import numpy as np
import pandas as pd
# 직접 구현
# Local alignment - alignment between highly relevant targets


def swaterman(s1, s2):
    x = len(s1) + 1  # keyword
    y = len(s2) + 1  # target

    mtx = np.zeros((x, y))  # similarity score matrix
    match = 3
    mismatch = 3
    gap = 2    # gap penalty

    for xi in range(0, x - 1):
        k = s1[xi]
        for yi in range(0, y - 1):
            if k == s2[yi]:
                score = max(mtx[xi][yi] + match, mtx[xi][yi + 1] - gap, mtx[xi + 1][yi] - gap, 0)
            else:
                score = max(mtx[xi][yi] - mismatch, mtx[xi][yi + 1] - gap, mtx[xi + 1][yi] - gap, 0)
            mtx[xi + 1][yi + 1] = score

    smax = np.max(mtx)  # find max value
    sx, sy = np.where(smax == mtx)  # find index

    keyword = []
    # find sequence from max values
    for i in range(0, len(sx)):
        tx, ty = sx[i], sy[i]
        temp = ""
        for o in range(ty, y - 1):
            temp += "-"

        #print(s1, s2)

        while True:
            t = np.argmax([mtx[tx-1][ty], mtx[tx][ty-1], mtx[tx-1][ty-1]])    # vertical, horizontal, match(diagonal)
            #print(tx, ty, " -- ", s2[ty - 1], " | ", t)

            if s1[tx - 1] == s2[ty - 1]:    # if same, add character and move diagonal
                temp = s2[ty - 1] + temp
                tx = tx - 1
                ty = ty - 1
            else:   # if different, move up or left then add black
                if t == 0:  # move up
                    tx = tx - 1
                elif t == 1:  # move left
                    ty = ty - 1
                temp = "-" + temp

            if mtx[tx][ty] <= 0:
                break

        for o in range(0, ty):
            temp = "-" + temp
        print(temp)

        keyword.append(temp)


"""
    mtx = pd.DataFrame(mtx).T

    print(mtx.loc(mtx["3"].idxmax()))

    print(mtx.idxmax())
#    print(mtx)
"""


if __name__ == "__main__":
    #swaterman("GGTTGACTA", "TGTTACGG")
    #swaterman("GET", "GET /index.html HTTP/1.0")
    #swaterman("HTTP/1.0", "GET /index.html HTTP/1.0")
    #swaterman(" /", "GET /index.html HTTP/1.0")
    #swaterman("POST", "GET /index.html HTTP/1.0")
    swaterman("GET  / HTTP/1.0", "GET /index.html HTTP/1.0")
    s1 = "GET /index.html HTTP/1.0"
    s2 = "POST"

    print(s1.find(s2))
    #
    exit(0)
