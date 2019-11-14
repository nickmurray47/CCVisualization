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
    distance_measure = util.euclidean_distance if sys.argv[
        2] == 'euclidean' else util.great_circle_distance
    filepath = sys.argv[3]

    # set a converge distance a bit arbitrarily
    converge_dist = 0.1

    # read in the data from the text file
    # get just the lat and lon data
    data = sc.textFile(filepath)
    lat_lon = data.map(lambda line: line.split()).map(
        lambda fields: (float(fields[0]), float(fields[1]))).persist()

    # initialize the means to distinct data points
    means = lat_lon.take(k)

    # start the big loop that ends when we reach convergence
    converged = False
    while not converged:
        pass


if __name__ == '__main__':
    main()
