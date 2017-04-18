import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def create_log_bin(lower_bound, upper_bound, bin_base):
    
    bin_edges = []
    bin_size = []
    
    bin_edges.append(lower_bound)
    edge = lower_bound 
    while edge <= upper_bound:
        edge *= bin_base
        bin_edges.append(edge)
        bin_size.append(bin_edges[-1] - bin_edges[-2])
        
    
    return np.array(bin_edges), np.array(bin_size)
    
    
def create_log_bin2(lower_bound, upper_bound, bin_num):
    
    bin_base = (upper_bound / lower_bound) ** (1. / bin_num)
    bin_edges = lower_bound * bin_base ** np.arange(bin_num+1)
    bin_edges[-1] = upper_bound
    bin_size = bin_edges[1:] - bin_edges[:-1]
    
    return bin_edges, bin_size


    

def cum_freq(x):
    
    x_sorted = sorted(x)
    num = [0]
    x_unique = [x_sorted[0]]
    
    for i in range(1,len(x_sorted)):
        if x_sorted[i] != x_sorted[i-1]:
            num.append(i)
            x_unique.append(x_sorted[i])
            
    return np.array(x_unique), np.array(num)
    


def E_cdf(x):
    
    x_unique, x_freq = cum_freq(x)
    cdf = x_freq / float(len(x))
    return x_unique, cdf
    
    

def E_ccdf(x):
    
    x_unique, x_freq = cum_freq(x)
    ccdf = 1.0 - x_freq / float(len(x))
    return x_unique, ccdf



def data_bin_stat(data, bin_edges):
    
    freq, bins = np.histogram(data, bin_edges)
    freq = np.float_(freq)
    bin_size = np.float_(bin_edges[1:] - bin_edges[:-1])    
    freq /= bin_size
    freq /= (sum(freq)+10**-10)
    
    return bin_edges[:-1], freq
    


def data_bin_stat2(data, bin_edges):
    
    freq, bins = np.histogram(data, bin_edges)
    freq = np.float_(freq)
    bin_size = np.float_(bin_edges[1:] - bin_edges[:-1])    
    freq /= bin_size
    
    return bin_edges[:-1], freq
    


def data_bin_stat3(idx, count, bin_edges, bin_size):
    
    freq = np.zeros(len(bin_edges)-1)

    for i in xrange(len(idx)):
        d = idx[i]
        ct = count[i]
        for j in xrange(len(bin_edges)):
            if d>=bin_edges[j] and d<=bin_edges[j+1]:
                freq[j] += ct

    freq /= bin_size
    freq /= (sum(freq)+10**-10)
    
    return bin_edges[:-1], freq



def data_bin_stat4(idx, count, bin_edges, bin_size):
    
    freq = np.zeros(len(bin_edges)-1)
    n = len(idx)
    emat = np.tile(bin_edges, (n,1))
    idxx = np.tile(idx, (len(bin_edges)-1, 1)).T
    t = np.logical_and( idxx >= emat[:,0:-1], idxx <= emat[:,1:] )

    idxs = np.nonzero(t)

    for i in xrange(len(idxs[1])):        
        freq[idxs[1][i]] += count[i]

    freq /= bin_size
    freq /= (sum(freq)+10**-10)
    
    return bin_edges[:-1], freq



def data_bin_XY(input_X, input_Y, bin_edges):
    
    output_Y = []
    output_X = []
    for i in range(1, bin_edges.size):
        index = np.logical_and(input_X < bin_edges[i], input_X >= bin_edges[i-1])
        if input_X[index].size >0:
            output_Y.append(input_Y[index].mean())
            output_X.append(bin_edges[i-1])
        del index
    
    return np.array(output_X), np.array(output_Y)
    
    

def data_bin_XY2(input_X, input_Y, bin_edges, operate_func = np.mean):
    
    output_Y = []
    output_X = []
    for i in range(1, bin_edges.size):
        index = np.logical_and(input_X < bin_edges[i], input_X >= bin_edges[i-1])
        if input_X[index].size > 0:
            output_Y.append(operate_func(input_Y[index]))
            output_X.append(bin_edges[i-1])
        del index
    
    return np.array(output_X), np.array(output_Y)
    
    
    
def double_linear_regression(data_X, data_Y, turning_point_id = None, start_id = 2, end_id = -3):
    
    n = len(data_X)
    if n < 5:
        print 'The length of the input data should be at least 3.'
        return -1
    
    if end_id < 0:
        end_id = n + end_id
    
    max_id = turning_point_id    
    if turning_point_id == None:
        
        rs_min = float(np.inf)
        
        for i in range(start_id, end_id+1):
            slope1_cur, intercept1_cur, r_value1, p_value1, std_err1 = stats.linregress(data_X[0:i+1], data_Y[0:i+1])
            slope2_cur, intercept2_cur, r_value2, p_value2, std_err2 = stats.linregress(data_X[i:], data_Y[i:])
            r_value1 *= r_value1
            r_value2 *= r_value2
            rs1 = ((data_Y[0:i+1] - np.polyval([slope1_cur, intercept1_cur], data_X[0:i+1])) ** 2).sum()
            rs2 = ((data_Y[i:] - np.polyval([slope2_cur, intercept2_cur], data_X[i:])) ** 2).sum()
            if rs1 + rs2 < rs_min:
                rs_min = rs1 + rs2
                max_id = i
                slope1 = slope1_cur
                slope2 = slope2_cur
                intercept1 = intercept1_cur
                intercept2 = intercept2_cur
            #print i, slope1_cur, slope2_cur, rs1 + rs2, std_err1, np.sqrt(rs1 / (i-1.)) * np.sqrt(1./ ((data_X[0:i+1] - data_X[0:i+1].mean()) ** 2).sum())
    
    else:
        if turning_point_id < 1 or turning_point_id > n-2:
            print 'The index of the turning point is not correct.'
            return -1
        else:
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(data_X[0:turning_point_id+1], data_Y[0:turning_point_id+1])
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(data_X[turning_point_id:], data_Y[turning_point_id:])
    
    #plt.plot(data_X, data_Y, 'o')
    #plt.plot(data_X[0:max_id+1], np.polyval([slope1, intercept1], data_X[0:max_id+1]), '-')
    #plt.plot(data_X[max_id:], np.polyval([slope2, intercept2], data_X[max_id:]), '-')
    #plt.show()
    
    return [slope1, intercept1, slope2, intercept2, max_id, rs_min]