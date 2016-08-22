import sys
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plot
# import pandas
import pandas as pd
import scipy
import sklearn.cluster as skc
from sklearn.cluster import DBSCAN

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import distance
from operator import itemgetter

from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import asksaveasfilename
from collections import Counter

# FVL (Frequent Visited Locations) generates the cluters from user's history locations and each cluster represents a user's top visited location
# Further, the ranks of the top locations are based on the number of points in each cluster
# Input: Trajectory file including the sorted trajectories from all users (each line represents a unique user)
# Output: ranked top visisted locations for each user, accompanyed with the purity measure (each line represents a unique user)

def FVL(pathIn, pathOut):
	print 'program running'
	if pathIn == '' or pathOut == '':
		mFile = open("chicago_2014_final_traj.txt","rb")
		mResultFile = open("top_locations.txt", "wb")
	else:
		mFile = open(pathIn, "rb")
		mResultFile = open(pathOut, "wb")

	for indx, mRow in enumerate(mFile):
		print 'processing ...' + str(indx)
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
			tweets_locations.append([float(lng), float(lat), mTimeStamp, mPolyID, mLandUse])

		if len(tweets_locations) > 4 and len(tweets_locations) < 2683:
			myarray = np.asarray(tweets_locations)
			df = pd.DataFrame(myarray)
			df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
			geo_positions = df[['lng', 'lat']]
			
			EPSILON = 0.0025 # this is NOT 500 meters, those are flat 0.005 degrees, which are ~555 meters (simpler)
			SAMPLES = 4  # this is the number of minimum samples (people), ten yields too many cluster, 50 seems good
			db  = skc.DBSCAN(eps=EPSILON, min_samples=SAMPLES)
			labels  = db.fit_predict(geo_positions.values)
			core_samples = db.core_sample_indices_
			labels = db.labels_
			# print labels
			df['cluster'] = labels
			# print df
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			# print n_clusters_

			grouped = df.groupby(['cluster'])

			result_temp = []

			for key, group in grouped:
				if key != -1.0:
					# print group
					ww = get_top_locations_from_cluster(group['landuse'])

					# print ww[0], ww[1]

					result_temp.append([key, group['cluster'].count(), ww[0], ww[1]])
			if result_temp != []:
				result = sorted(result_temp, key=itemgetter(1), reverse=True)
				print result_temp
				mOutput = ''
				for mComponents in result:
					mOutput += mComponents[2] + ":" + str(mComponents[3]) + ";" 
				mResultFile.write(mOutput)
				mResultFile.write("\n")
	
	mResultFile.close()

	# The following line of code is only for visualization purpose
	# 		import pylab as pl
	# 		from itertools import cycle

	# 		pl.close('all')
	# 		pl.figure(1)
	# 		pl.clf()

	# 	# Black removed and is used for noise instead.
	# 		colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
	# 		for k, col in zip(set(labels), colors):
	# 			if k == -1:
	# 			# Black used for noise.
	# 				col = 'k'
	# 				markersize = 6
	# 			class_members = [index[0] for index in np.argwhere(labels == k)]
	# 			cluster_core_samples = [index for index in core_samples if labels[index] == k]
	# 			for index in class_members:
	# 				x = myarray[index]
	# 				if index in core_samples and k != -1:
	# 					markersize = 14
	# 				else:
	# 					markersize = 6
	# 				pl.plot(x[0], x[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=markersize)
	# 		pl.title('Estimated number of clusters: %d' % n_clusters_)
	# 		pl.show()
	# mResultFile.close()

##this function determines the dominate "land use" type in each cluster and returns the dominate land use type as well as the purity measure
def get_top_locations_from_cluster(cluster):
	words_to_count = (word for word in cluster if word)
	c = Counter(words_to_count)
	mMostCount = c.most_common()[:1]
	return [mMostCount[0][0], (mMostCount[0][1]*1.0) /(len(cluster) * 1.0)]

if __name__ == '__main__':
	Tk().withdraw() # keep the root window from appearing
	openFileName = askopenfilename() # show an "Open" dialog box and return the path to the selected file
	saveFileName = asksaveasfilename()
	FVL(openFileName, saveFileName)