from scipy.spatial import distance
from math import radians, degrees, sin, cos, asin, sqrt, atan2


r = 6371


# given point and a list of centers, outputs the index of the center
# closest to the point using the given distance measure
def closest_point(point, centers, get_distance):
    distances = [get_distance(point, center) for center in centers]
    return min(enumerate(distances), key=lambda distance: distance[1])[0]


def add_points(point1, point2):
    return tuple([point1[i] + point2[i] for i in range(len(point1))])


# takes lat/lon point and outputs a x, y, z cartesian representation
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def geo_to_cartesian(point):
    lat, lon = map(radians, point)

    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)

    return x, y, z


# takes in cartesian coordinates and outputs lat/lon point
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
def cartesian_to_geo(x, y, z):
    lat = asin(z / r)
    lon = atan2(y, x)

    return (degrees(lat), degrees(lon))


# returns the euclidean distance between 2 points
def euclidean_distance(point1, point2):
    return distance.euclidean(point1, point2)


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
