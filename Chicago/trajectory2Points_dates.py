import sys
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plot

import numpy as np
from operator import itemgetter
import pandas as pd
import datetime

# from Tkinter import Tk
# from tkFileDialog import askopenfilename
# from tkFileDialog import asksaveasfilename

## Break all the trajectories into points
## Input: Trajectory file
## Output: a csv file containing all the tweets with the additional information: hour of the day, day of the week
def tweets_with_dates():
	print 'coverting tweets with dates'
	# openFileName = askopenfilename()
	mFile = open("C:\\project\\sq_mobility\\chitown\\2014_all_chi_traj.txt", "rb")
	# mFile = open("chicago_2014_final_traj.txt","rb")

	for mRow in mFile:
		tweets_locations = []
		mTime = []
		mLandUseC = []
		mPolyIDC = []
		mTraj = mRow.replace('\n', '').split(',')
		
		for mLoc in mTraj[1:]:
			xx = []
			[lng, lat, mTimeStamp, mPolyID, mLandUse] = mLoc.split('&')
			xx.append(float(lng))
			xx.append(float(lat))
			tweets_locations.append([float(lng), float(lat), int(mTimeStamp)/1000, mPolyID, mLandUse])

		
		myarray = np.asarray(tweets_locations)
		df = pd.DataFrame(myarray)
		df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
		geo_positions = df[['lng', 'lat']]
		df['timestamp'] = df['timestamp'].astype(int)
		
		##Deal with the daylight saving changes
		for ff in df['timestamp']:
			xx = datetime.datetime.fromtimestamp(ff).strftime('%m-%d')
			if xx > '03-20' and xx < '11-00':
				# print 'cst'
				df.loc[df['timestamp'] == ff, 'timestamp'] = ff - 3600 * 5
			else:
				df.loc[df['timestamp'] == ff, 'timestamp'] = ff - 3600 * 6
		df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')		

		with open("C:\\project\\sq_mobility\\chitown\\chicago_2014_all_dates.csv", "wb") as myfile:
				df.to_csv(myfile, sep='\t', mode='a', header=False, index=False)
				


if __name__ == '__main__':
	tweets_with_dates()

