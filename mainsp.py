import time
import os
import sys

from src.functions import idf_dic, tf_idf, tf_dic, cosine_sim

__author__ = 'Petr Vecera'


def load_file(file_name):
    """
    Load single file as a string
    :param file_name:
    :return: string with all the file data
    """
    try:
        f = open(file_name, "r", encoding="utf8")
    except Exception as e:
        print(e)
        exit(1)
    else:
        return f.read()


def load_data(path_to_file):
    """

    :param path_to_file: Path to file  which have each line as path to data file
    :return: List of strings (data files)
    """
    start_time = time.time()
    try:
        f = open(path_to_file, "r", encoding="utf8")
    except Exception as e:
        print(e)
        exit(1)

    list_of_text_files = f.readlines()
    list_od_data = []
    for x in list_of_text_files:
        list_od_data.append(load_file(x[:len(x)-1]))
    print("Data load time\t\t--- {:.3f} seconds ---".format(time.time() - start_time))
    print("Loaded files:")
    for i, x in enumerate(list_of_text_files):
        print("File {}: {}".format(i+1, os.path.basename(x)), end="")
    print("")

    return list_od_data


def start_analyze(list_of_text):
    s_time = time.time()
    tfdiclist = []

    for x in list_of_text:
        tfdiclist.append(tf_dic(x))

    tf_time = time.time() - s_time
    s_time = time.time()

    idf = idf_dic(tfdiclist)

    idf_time = time.time() - s_time
    s_time = time.time()

    tfidflist = []
    for x in tfdiclist:
        tfidflist.append(tf_idf(x, idf))

    tfidf_time = time.time() - s_time
    s_time = time.time()

    print("")
    for i, x in enumerate(tfidflist):
        for z, y in enumerate(tfidflist):
            print("Text {}-{}: {:.3f}".format(i+1, z+1, cosine_sim(x, y)))
        print("")

    csim_time = time.time() - s_time

    print("TF compute time\t\t--- {:.3f} seconds ---".format(tf_time))
    print("IDF compute time \t--- {:.3f} seconds ---".format(idf_time))
    print("TF*IDF compute time \t--- {:.3f} seconds ---".format(tfidf_time))
    print("Cos Sim compute time\t--- {:.3f} seconds ---".format(csim_time))


def main():
    start_time = time.time()

    filename = "filelist.txt"

    if len(sys.argv) <= 1:
        print("Input file not specified trying with default:\n"
              "Usage: file.py filelist.txt")
    else:
        filename = sys.argv[1]

    data = load_data(filename)
    start_analyze(data)

    print("Complete compute time\t--- {:.3f} seconds ---".format(time.time() - start_time))
    return 0


if __name__ == '__main__':
    main()
