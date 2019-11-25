import sys
from pyspark import SparkContext
from . import util


def main():
    # check to make sure command line arguments are passed properly
    if sys.argv != 5:
        print('Usage: k_means.py <k> <distance_measure> <input> <output>',
              file=sys.stderr)
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
    lat_lon = data.map(lambda line: line.split()).map(
        lambda fields: (float(fields[0]), float(fields[1]))).persist()

    # initialize the means to distinct data points
    means = lat_lon.takeSample(False, k)

    # start the big loop that ends when we reach convergence
    converged = False
    while not converged:
        assignments = lat_lon.map(lambda point: (util.closest_point(
            point, means, distance_measure), point))

        if distance_measure == 'euclidean':
            new_means_rdd = assignments.groupByKey().map(
                lambda i, points: (i, util.average_points(points)))
        else:
            new_means_rdd = assignments.map(
                lambda i, point: (i, util.geo_to_cartesian(point)))\
                .groupByKey()\
                .map(lambda i, points: (i, util.average_points(points)))\
                .map(lambda i, point: (i, util.cartesian_to_geo(point)))

        new_means = [(0, 0) for _ in range(k)]
        for i, point in new_means_rdd.collect():
            new_means[i] = point

        converged = util.average_differences(means, new_means) < converge_dist
        means = new_means

    output_rdd = sc.parallelize(means).union(assignments)
    output_rdd.saveAsTextFile(output_path)


if __name__ == '__main__':
    main()
