
import numpy as np
def common_sequences(w1, w2):
    l1 = len(w1)
    l2 = len(w2)
    table = np.zeros([l1, l2], dtype=int)
    
    for i in range(l1):
        for j in range(l2):
            if w1[i] == w2[j]:
                table[i, j] = 1 if i==0 or j==0 else table[i-1, j-1] + 1
    #print(table)
    marker1 = np.zeros([l1])
    marker2 = np.zeros([l2])

    indices = []
    for i in reversed(range(l1)):
        for j in reversed(range(l2)):
            if table[i, j] > 0 and (marker1[i] == 0 or marker2[j] == 0):
                s1 = i - table[i, j] + 1
                e1 = i
                s2 = j - table[i, j] + 1
                e2 = j
                indices.append([s1, e1, s2, e2])
                marker1[s1:e1+1] = marker2[s2:e2+1] = 1

    return indices


def get_overlaps(w1, w2, p1, p2):
    
    l1 = len(w1) - 1
    l2 = len(w2) - 1
    ri1 = np.argmax(p1[l1, :])
    ri2 = np.argmax(p2[l2, :])
    
    overlaps = [[0]] * (l1 + l2 + 3)

    mx = 0
    candidates = []
    for s1, e1, s2, e2 in common_sequences(w1, w2):
        if s1 > 0 and e2 < l2:
            if e1 < l2 - s2:
                ci1 = e1 + 1
                ci2 = e2 + 1
            else:
                ci1 = s1
                ci2 = s2

            if mx < e1 - s1:
                mx = e1 - s1
                candidates = []

            if mx == e1 - s1:
                candidates.append((s1, e1, s2, e2, abs(ci1 - ri1) + abs(ci2 - ri2)))

    if len(candidates) > 0:
        s1, e1, s2, e2, _ = min(candidates, key=lambda item: item[2])
        #print(s1, e1, l1, s2, e2, l2)
        overlaps = [[0]] * s1 + [[1]] * (e1 - s1 + 1) + [[0]] * (l1 - e1)
        overlaps += [[0]]
        overlaps += [[0]] * s2 + [[1]] * (e2 - s2 + 1) + [[0]] * (l2 - e2)
        #print(overlaps)
        
    return overlaps


def test_common_sequences(w1, w2):
        
    for s1, e1, s2, e2 in common_sequences(w1, w2):
        print(s1, e1, s2, e2)
        print(w1[:e1 + 1] + w2[e2 + 1:])


if __name__ == "main":
    test_common_sequences('group', 'coupon')
# test_common_sequences('abc', 'xyz')
# test_common_sequences('obomo', 'monio')