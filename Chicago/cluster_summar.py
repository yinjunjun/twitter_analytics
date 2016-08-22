## Note: function cluster_summary requires Numpy 1.9.2

import sys
import numpy as np

def cluster_summary():
	openFileName = askopenfilename()
	mFile = open(openFileName, "rb")
	mOutputFVL = open("result/number_of_fvl.txt", "wb")

	# #data format: 1100:0.754385964912;1215:0.363636363636;1100:0.941176470588;
	
	for row in mFile:
		print '----------------'
		unique_landuse = []
		record = row.replace(";\n", "").split(";")
		# print record
		total_num_cluster = len(record)
		mOutputFVL.write(str(total_num_cluster) + '\n')

		for mm in record:
			[landuse, purity] = mm.split(":")
			unique_landuse.append(landuse)
		uniq_landuse, counts = np.unique(unique_landuse, return_counts=True)
		# print uniq_landuse
		
		temp_result = np.asarray((uniq_landuse, counts)).T
		print temp_result

		for uniq in temp_result:
			with open("result/landuse_" + uniq[0] + ".csv", "a") as myfile:
				myfile.write(str(total_num_cluster) + ',' + uniq[1] + '\n')
		myfile.close()

	mOutputFVL.close()


if __name__ == '__main__':
	Tk().withdraw() # keep the root window from appearing
	cluster_summary()
