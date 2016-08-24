import numpy as np

def realEntropy(lst):
    
    if len(set(lst)) == 1:
        return 0.0
        
    alpha = 0
    n = len(lst)
    sq = ''
    for i in range(n):
        sq += chr(32+int(lst[i]))
        
    for i in range(n):
        for j in range(i,n):
            if not (sq[i:j+1] in sq[:i]):
                alpha += j - i + 1
                break

    return (n * np.log2(n)) / alpha
            
            
                                    