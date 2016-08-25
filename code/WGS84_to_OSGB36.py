from osgeo import ogr
from osgeo import osr

def getXY(lat, lng):
	source = osr.SpatialReference()
	source.ImportFromEPSG(4326)

	target = osr.SpatialReference()
	target.ImportFromEPSG(27700)

	transform = osr.CoordinateTransformation(source, target)
	
	point = ogr.CreateGeometryFromWkt("POINT (%s %s)" %(lng,lat))
	point.Transform(transform)
	[x,y] = point.ExportToWkt().replace("POINT (",'').replace(')','').split(' ')
	return [x,y]


# print getXY(55.4, -2.89)

def main():
	print "project running"
	mOutputFile = open("uk_nov_proj.txt", "wb")
	with open("uk_nov.txt") as mFile:
		for line in mFile:
			# 165831846,53.544955,-2.767532,1401598777000
			content = line.split(',')
			lat = content[1]
			lng = content[2]
			[x, y] = getXY(lat, lng)
			mOutputFile.write(content[0] + ',' + x + ',' + y + ',' + content[3])
	mOutputFile.close()


        




if __name__ == '__main__':
	main()
