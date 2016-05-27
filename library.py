import numpy
import math

def l_2(w):
    return math.sqrt(sum([val ** 2 for val in w]))

def avg(l):
    return float(sum(l))/len(l)
    
default_output_file = "output"
def output_final_w(wstar):
    f = open(default_output_file,'w')
    wstarconfig = "//w* is " + str(wstar)
    wstarconfig = wstarconfig.replace("\n", "")
    wstarconfig = wstarconfig.replace(",", "")
    f.write(wstarconfig)

def load_data(filename):
    f = open(filename, "r")
    print "Reading from file " + filename
    # Data from the UniformDataGenerator.py has the format
    # such that the first line is w*
    line = f.readline()
    line = line[line.index('[') + 1 : len(line) - 2]
    # Remove the other comment part of the first line of w*
    tokens = line.split()
    wstar = []
    for token in tokens:
        wstar.append(float(token))

    print "W* is " + str(wstar)

    ds = []
    for line in f:
        features = []
        sign = 0
        tokens = line.split()
        for i in range(0, len(tokens) - 1):
            features.append(float(tokens[i]))
        sign = int(tokens[len(tokens) -1])
        ds.append((features,sign))
    print "Done loading data"
    return (wstar,ds)

def calc_error_rate(wstar, dataset):
    return float(count_errors(wstar, dataset))/ float(len(dataset))

def count_errors(wstar, dataset):
    errors = 0
    num_samples = len(dataset)
    for (features, label) in dataset:
        dp = numpy.dot(features, wstar)
        if dp > 0 and label == -1:
            errors = errors + 1
        elif dp < 0 and label == 1:
            errors = errors + 1
        elif dp == 0 and label == 1:
            # <= 0 -> -1, so 1 would be an error
            errors = errors + 1
    return errors

def dot_product(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))