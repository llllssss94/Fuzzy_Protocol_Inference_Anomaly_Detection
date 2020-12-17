# Module Import
from apriori import extract
from nwunch import swunch
from warterman import swaterman

import skfuzzy as fuzz
import numpy as np
import matplotlib.pylab as plt


def dynamicCal(key, target):
    kl = key.split(" ")
    rule = []
    i = 0

    for k in kl:
        offset = target.find(k)
        if offset < 0:  # If there is no keyword in target,
            return

        if len(k) <= 0:  # Noise canceling
            continue

        if len(rule) > 0:
            rule.append({"len": offset - rule[i - 1]["len"], "type": i})    # dynamic field size
            rule.append({"len": offset + len(k) + 1, "type": k})    # keyword end position
        else:
            rule.append({"len": offset + len(k) + 1, "type": k})    # keyword end position
        i += 1

    return rule

# Extract static keywords from sample data
keywords = extract()
ruleset = []

s = "GET /webtoon/op/ff455fe7f3cf441c9b5ea13500c7e3d09b6240c8 HTTP/1.1"
for key in keywords:
    rule = dynamicCal(key, target=s)
    if rule is not None:
        ruleset.append((rule, s))

print(ruleset)


"""
# Fuzzy Part
x = np.array([[2, 3, 1, 2], [2, 1, 1, 0]])

n_sample, n_features = x.shape

center, mem, _, _, _, _, _ = fuzz.cmeans(x.T, 2, 2.0, error=1e-5, maxiter=200)

delta = np.zeros([2, n_features])

for i in range(2):
    d = (x - center[i, :]) **2
    print(d)
"""

exit(0)