import numpy


def calculate_tweetNumbers():
    print "calculating Tweet numbers of people ..."
    mFile = open("C:\\project\\sq_mobility\\chitown\\2014_all_chi_traj.txt","rb")
    
    pFile = open("C:\\project\\sq_mobility\\chitown\\tweetCount_Chicago_2014.txt", "wb")
    
    for mLine in mFile:
        line = mLine.split(',')
        mNumber = len(line)
        if mNumber > 1:
            userid = line[0]
            pFile.write(str(userid) + "," + str(mNumber -1) + '\n')
    
    pFile.close()
    
    


if __name__ == "__main__":
    print "running"
    calculate_tweetNumbers()
    print "finished"