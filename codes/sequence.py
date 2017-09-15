# import numpy as np

# ww = np.array([1,2,3,3,3,3,2,2,2,1,1,2,2,3,3])

# mm = []

# mFirst = ww[0:-1]
# mLast = ww[1:]

# # print mLast - mFirst
# # for i, item in enumerate(ww):
# # 	if ww[]

# mm.append(mFirst[0])
# for xx in range(len(ww) - 1):
# 	# mm.append(ww[])
# 	if ww[xx] != ww[xx+1]:
# 		mm.append(ww[xx+1])

# # if ww[-1] != mm[-1]:
# # 	mm.append(ww[-1])

# print mm

# mCluster=[1,1,2,3,3,2,1]

# uniq_locations, indices = np.unique(mCluster, return_inverse=True)

# print uniq_locations


import numpy as np

def realEntropy(lst):
    
    if len(set(lst)) == 1:
        return 0.0
        
    alpha = 0
    # n = len(lst)
    n=len(set(lst))
    sq = ''
    for i in range(n):
        sq += chr(32+int(lst[i]))
        
    for i in range(n):
        for j in range(i,n):
            if not (sq[i:j+1] in sq[:i]):
                alpha += j - i + 1
                break

    return (n * np.log2(n)) / alpha
            

def uncEntropy(lst):
    uniq_locations, counts = np.unique(lst, return_counts=True)
    # temp_result = np.asarray((uniq_locations, counts)).T
    mP =  (counts*1.0)/(sum(counts)*1.0)
    mEntropy_temp = mP * np.log2(mP)
    mEntropy = -np.sum(mEntropy_temp)
    return mEntropy


def randEntropy(lst):
    uniq_locations, counts = np.unique(lst, return_counts=True)
    # temp_result = np.asarray((uniq_locations, counts)).T
        #print uniq_locations
        #print counts
        #print temp_result
    mP =  (counts*1.0)/(sum(counts)*1.0)
    mEntropy = np.log2(len(uniq_locations))
    return mEntropy



w1 = [12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 1, 14, 14, 14, 14, 4, 4, 3, 4, 4, 2, 7, 6, 8, 5, 9, 11, 0, 10, 13, 14, 14, 14, 14, 14, 14, 14, 14]
w2 = [12, 14, 1, 14, 4, 3, 4, 2, 7, 6, 8, 5, 9, 11, 0, 10, 13, 14]

x1 = realEntropy(np.array(w1))
x2 = realEntropy(np.array(w2))
x3 = uncEntropy(np.array(w1))
x4 = randEntropy(np.array(w1))

# x4 = uncEntropy(np.array(w2))
print x1
print x2
print x3
print x4


