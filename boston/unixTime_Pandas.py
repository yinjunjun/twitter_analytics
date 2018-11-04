import sys
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plot

import numpy as np
from operator import itemgetter
import pandas as pd
import datetime

## Input: Trajectory file
## Output: a csv file containing all the tweets with the additional information: hour of the day, day of the week
def tweets_with_dates():
    print 'coverting tweets with dates'
    mFile = open("testdata.txt", "rb")

    for mRow in mFile:
        tweets_locations = []
        mTime = []
        mLandUseC = []
        mPolyIDC = []
        [uid, lat, lng, mTimeStamp, mTimeText] = mRow.replace('\n', '').split(',')
        # uid = mTraj[0]

        tweets_locations.append([uid,float(lng), float(lat), int(mTimeStamp)/1000, mTimeText])
        myarray = np.asarray(tweets_locations)
        df = pd.DataFrame(myarray)
        df.columns = ['uid','lat', 'lng', 'timestamp', 'mTimeText']
        #geo_positions = df[['lng', 'lat']]
        df['timestamp'] = df['timestamp'].astype(int)

		##Deal with the daylight saving changes
        for ff in df['timestamp']:
            print ff
            yy = datetime.datetime.fromtimestamp(ff)
            xx = datetime.datetime.fromtimestamp(ff).strftime('%m-%d')
            print yy
            # print xx
            if xx > '03-20' and xx < '11-00':
                print 'xx'
                df.loc[df['timestamp'] == ff, 'timestamp'] = ff - 3600 * 5
            else:
                print 'yy'
                df.loc[df['timestamp'] == ff, 'timestamp'] = ff - 3600 * 6
                # print datetime.datetime.fromtimestamp(ff - 3600 * 5)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        print df

        # for ff in df['timestamp']:
        #     print ff
        #     yy = datetime.datetime.fromtimestamp(ff)
        #     print yy
        #with open("../result/chicago_2014_all_dates_unix.csv", "wb") as myfile:
        #    df.to_csv(myfile, sep='\t', mode='a', header=False, index=False)

if __name__ == '__main__':
    tweets_with_dates()

