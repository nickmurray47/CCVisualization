from pyspark import SparkContext


def main():
    # setup spark context
    sc = SparkContext(appName='Data Scrubber')

    # read in data
    data = sc.textFile(
        'file:/home/cloudera/training_materials/dev1/data/devicestatus.txt')

    # scrub data for valid entries
    valid_data = data.map(lambda line: line.split(line[19])).filter(
        lambda tokens: len(tokens) == 14)

    # scrub the valid data for the date, model, device id, latitude,
    # and longitude
    # then scrub out the entries with lat and lon of 0
    relevant_fields = valid_data.map(
        lambda tokens: (tokens[12], tokens[13], tokens[0],
                        tokens[1].split(' ')[0], tokens[1].split(' ')[1],
                        tokens[2])).filter(
        lambda fields: fields[0] != '0' or fields[1] != '0')

    # format the data into csv format then save it out
    csv_format = relevant_fields.map(lambda fields: " ".join(fields))
    csv_format.coalesce(1, shuffle=True).saveAsTextFile(
        '/loudacre/devicestatus_etl')

    sc.stop()


if __name__ == '__main__':
    main()
