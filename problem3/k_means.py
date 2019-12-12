import sys
from pyspark import SparkContext
from math import radians, degrees, sin, cos, asin, sqrt, atan2


r = 6371


# given point and a list of centers, outputs the index of the center
# closest to the point using the given distance measure
def closest_point(point, centers, distance_measure):
    get_distance = euclidean_distance if distance_measure == 'euclidean' \
        else great_circle_distance
    distances = [get_distance(point, center) for center in centers]
    return min(enumerate(distances), key=lambda distance: distance[1])[0]


# takes in a list of points
# outputs the average point
def average_points(points):
    avg_point = [0 for _ in range(len(list(points)[0]))]
    for point in points:
        for i, attr in enumerate(point):
            avg_point[i] += attr

    avg_point = [attr / len(points) for attr in avg_point]
    return tuple(avg_point)


# takes lat/lon point and outputs a x, y, z cartesian representation
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def geo_to_cartesian(point):
    lat, lon = map(radians, point)

    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)

    return (x, y, z)


# takes in cartesian coordinates and outputs lat/lon point
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def cartesian_to_geo(point):
    x, y, z = point
    lat = asin(z / r)
    lon = atan2(y, x)

    return (degrees(lat), degrees(lon))


# returns the euclidean distance between 2 points
def euclidean_distance(point1, point2):
    diff_squared = [(elem1 - point2[i]) ** 2 for i, elem1 in enumerate(point1)]
    return sqrt(sum(diff_squared))


# returns the great circle distance between 2 points
# https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
def great_circle_distance(point1, point2):
    lat1, lon1 = map(radians, point1)
    lat2, lon2 = map(radians, point2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * \
        cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = r * c
    return km


# takes in 2 arrays of points
# outputs average element-wise distances between the 2 arrays
def average_differences(arr1, arr2, distance_measure):
    get_distance = euclidean_distance if distance_measure == 'euclidean' \
        else great_circle_distance
    distances = [get_distance(elem1, arr2[i]) for i, elem1 in enumerate(arr1)]
    return sum(distances) / len(distances)


def main():
    # check to make sure command line arguments are passed properly
    if len(sys.argv) != 5:
        print('Usage: k_means.py <k> <distance_measure> <input> <output>')
        exit(-1)

    # initialize spark context
    sc = SparkContext(appName='k means')

    # initialize the variables related to the command line arguments
    k = int(sys.argv[1])
    distance_measure = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    # set a converge distance a bit arbitrarily
    converge_dist = 0.1

    # read in the data from the text file
    # get just the lat and lon data
    data = sc.textFile(input_path)
    # lat_lon = data.map(lambda line: line.split())\
    #     .filter(lambda fields: fields)\
    #     .map(lambda fields: (float(fields[0]), float(fields[1]))).persist()
    lat_lon = data.map(lambda line: line.split(','))\
        .filter(lambda fields: fields)\
        .map(lambda fields: (float(fields[1]), float(fields[2]))).persist()

    # initialize the means to distinct data points
    means = lat_lon.takeSample(False, k) #how we randomly select the means

    # start the big loop that ends when we reach convergence
    converged = False
    while not converged:
        # get the mean point assignment for each point
        assignments = lat_lon.map(lambda point: (closest_point(
            point, means, distance_measure), point))

        # calculate the new mean based on the assignment using the
        # relevant distance measure
        if distance_measure == 'euclidean':
            new_means_rdd = assignments.groupByKey().map(
                lambda line: (line[0], average_points(line[1])))
        else:
            new_means_rdd = assignments.map(
                lambda line: (line[0], geo_to_cartesian(line[1])))\
                .groupByKey()\
                .map(lambda line: (line[0], average_points(line[1])))\
                .map(lambda line: (line[0], cartesian_to_geo(line[1])))

        # output the new means as a python array
        new_means = [(0, 0) for _ in range(k)]
        for line in new_means_rdd.collect():
            new_means[line[0]] = line[1]

        # check for convergence
        converged = average_differences(
            means, new_means, distance_measure) < converge_dist
        means = new_means

    # format the output and output it
    output_rdd = sc.parallelize(enumerate(means))\
        .map(lambda line: 'c %d %f %f' % (line[0], line[1][0], line[1][1]))\
        .union(assignments.map(
            lambda line: 'p %d %f %f' % (line[0], line[1][0], line[1][1])))
    output_rdd.coalesce(1, shuffle=True).saveAsTextFile(output_path)

    sc.stop()


if __name__ == '__main__':
    main()
