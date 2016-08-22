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

##Get the temporal signatures of each land use type at each specific rank
# Input: Trajectory file
# Output: CSV files organized by the rank of clusters, e.g. points_1.csv contains all the top 1 clusters (with all the points in the cluster)
import datetime
def temporal_distribution():
	print 'temporal_distribution'
	print 'program running'
	# mFile = open("chicago_jan_traj3.txt","rb")
	mFile = open("chicago_2014_final_traj.txt","rb")

	# mResultFile2 = open("top_locations_temporal.txt", "wb")
	
	extension = 1
	
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
			tweets_locations.append([float(lng), float(lat), int(mTimeStamp), mPolyID, mLandUse])

		if len(tweets_locations) > 4 and len(tweets_locations) < 2683:
		#if len(tweets_locations) == 1000:
			myarray = np.asarray(tweets_locations)
			# myarray1 = np.asarray(mTime)

			df = pd.DataFrame(myarray)
			df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
			geo_positions = df[['lng', 'lat']]

			df['timestamp'] = df['timestamp'].astype(int)

			# print df

			for ff in df['timestamp']:
				
				xx = datetime.datetime.fromtimestamp(ff/1000).strftime('%m-%d')
				# print xx
				if xx > '03-20' and xx < '11-00':
					# print 'cst'
					df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 5
				else:
					df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 6
			df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')


			EPSILON = 0.0025 # this is NOT exactly 250 meters, those are flat 0.005 degrees
			SAMPLES = 4 #int(len(tweets_locations) * 0.05)  # this is the number of minimum samples (people), ten yields too many cluster, 50 seems good
			db  = skc.DBSCAN(eps=EPSILON, min_samples=SAMPLES)
			labels  = db.fit_predict(geo_positions.values)
			core_samples = db.core_sample_indices_
			labels = db.labels_
			# print labels
			df['cluster'] = labels
			# print df
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			# print n_clusters_
			with open("points" + str(extension) + ".csv", "wb") as myfile:
				df.to_csv(myfile, sep='\t', mode='a', header=False, index=False)
			
			extension +=1

			grouped = df.groupby(['cluster'])

			result_temp = []

			for key, group in grouped:
				if key != -1.0:
					# print group
					ww = get_top_location_from_cluster(group['landuse'])

					# print ww[0], ww[1]

					result_temp.append([key, group['cluster'].count(), ww[0], ww[1]])
			
			if result_temp != []:
				result = sorted(result_temp, key=itemgetter(1), reverse=True)
				# print result

				for key2, group2 in grouped:
					# print key2, group2
					if key2 != -1.0:
						# print group2
						for index, mKey in enumerate(result):
							if key2 == int(mKey[0]):
								print index
								print 'Top cluster # '+ str(mKey[0])
								temp_result = group2.loc[group2['landuse'] == mKey[2]]
								print temp_result[['timestamp', 'landuse']]
								# for mElements in temp_result['timestamp', 'landuse']:
								# 	print mElements

								
								with open("result" + str(index) + ".csv", "a") as myfile:
								 	df.to_csv(myfile, sep='\t', mode='a', header=False, index=False, columns=['timestamp', 'landuse'])


from collections import Counter
def get_top_location_from_cluster(cluster):
	words_to_count = (word for word in cluster if word)
	c = Counter(words_to_count)
	mMostCount = c.most_common()[:1]
	return [mMostCount[0][0], (mMostCount[0][1]*1.0) /(len(cluster) * 1.0), cluster]
		

if __name__ == '__main__':
	temporal_distribution()
	# cluster_summary()
