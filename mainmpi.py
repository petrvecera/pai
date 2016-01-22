import time
from mpi4py import MPI
import sys
import os

from src.functions import idf_dic, tf_idf, tf_dic, cosine_sim

__author__ = 'Petr Vecera'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()


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
        kill_slaves()
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
        kill_slaves()
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
    """
    Controls all the computing. And prints out the results.
    :param list_of_text: List of string, each string is one text file
    :return:
    """
    s_time = time.time()
    tfdiclist = []

    # HERE SEND TO CLIENTS for TF compute
    for i, x in enumerate(list_of_text):
        comm.send("tf", dest=i+1, tag=0)
        comm.send(x, dest=i+1, tag=1)

    for i, x in enumerate(list_of_text):
        data = comm.recv(source=i+1, tag=2)
        tfdiclist.append(data)

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
        comm.send("cs", dest=i+1, tag=0)
        comm.send(tfidflist, dest=i+1, tag=3)

    for i, x in enumerate(tfidflist):
        data = comm.recv(source=i+1, tag=3)
        print(data)

    csim_time = time.time() - s_time

    print("TF compute time\t\t--- {:.3f} seconds ---".format(tf_time))
    print("IDF compute time\t--- {:.3f} seconds ---".format(idf_time))
    print("TF*IDF compute time\t--- {:.3f} seconds ---".format(tfidf_time))
    print("Cos Sim compute time\t--- {:.3f} seconds ---".format(csim_time))


def kill_slaves():
    """
    Kills all the waiting processes, should be run before quiting the program.
    :return:
    """
    for x in range(1, rank_size):
        comm.send("ks", dest=x, tag=0)


def master():
    """
    Master MPI process, which handles all the control
    :return:
    """

    start_time = time.time()

    filename = "filelist.txt"
    if len(sys.argv) <= 1:
        print("Input file not specified trying with default:\n"
              "Usage: file.py filelist.txt")
    else:
        filename = sys.argv[1]

    data = load_data(filename)
    if len(data) >= rank_size:
        print("Error we need at n+1 threads, where n are text files.\n"
              "Got: {} files, but only {} threads".format(len(data), rank_size))
        kill_slaves()
        exit(1)
    start_analyze(data)

    print("Complete compute time\t--- {:.3f} seconds ---".format(time.time() - start_time))


def slave():
    """
    Slave MPI process, waits for jobs names until killed by command "ks"
    :return:
    """
    kill_slaves = False
    while not kill_slaves:
        job_name = comm.recv(source=0, tag=0)

        if job_name == "ks":
            kill_slaves = True

        elif job_name == "tf":
            data = comm.recv(source=0, tag=1)
            data = tf_dic(data)
            comm.send(data, dest=0, tag=2)

        elif job_name == "cs":
            data = comm.recv(source=0, tag=3)
            my_str = ""
            for z, y in enumerate(data):
                my_str += "Text {}-{}: {:.3f}\n".format(rank, z+1, cosine_sim(data[rank-1], y))
            comm.send(my_str, dest=0, tag=3)


def main():
    if rank == 0:
        master()
        kill_slaves()
    else:
        slave()

    return 0

if __name__ == '__main__':
    main()
