import re
import math

__author__ = 'Petr Vecera'


def norm_dict(w_dict, length):
    """
    Normalizes dict with the input length.
    :param w_dict:
    :param length:
    :return: dictioanry
    """
    for y in w_dict:
        w_dict[y] /= length


def tf_dic(data):
    """
    Calculates the TF - Term Frequency
    :param data: String with the document.
    :return: Dictionary with words and number of occurrence.
    """
    dic = {}
    # Remove unwanted chars and also lower all the words
    reg = re.compile('[^a-zA-Z]')
    data = data.lower()
    data = reg.sub(' ', data)
    data = data.split()
    len_of_all_words = len(data)
    for x in data:
        if x in dic.keys():
            dic[x] += 1
        else:
            dic[x] = 1

    norm_dict(dic, len_of_all_words)
    return dic


def idf_for_term(term, list_of_dic):
    """
    Calculates the IDF for one single term. (How much is term common)
    :param term: One term, string.
    :param list_of_dic: List of dictionaries (number of documents)
    :return: Number - IDF
    """
    term_used = 0
    for x in list_of_dic:
        if term in x.keys():
            term_used += 1
    if term_used > 0:
        return 1 + math.log((len(list_of_dic)/term_used), math.e)
    else:
        return 1


def idf_dic(list_of_dic):
    """
    Creates IDF dictionary - all the words in the input dictionaries
    :param list_of_dic:
    :return: dictionary
    """
    # Let's copy the first dic
    z = list_of_dic[0].copy()
    # Let's merge all the dic into a one
    for x in list_of_dic:
        z.update(x)

    # Create IDF dictionary
    for key in z:
        z[key] = idf_for_term(key, list_of_dic)
    return z


def tf_idf(tf_dic, idf_dic):
    """
    Crates the TF*IDF dictionary
    :param tf_dic:
    :param idf_dic:
    :return: dictionary
    """
    z = tf_dic.copy()
    for x in tf_dic:
        z[x] = tf_dic[x] * idf_dic[x]
    return z


def cosine_sim(dic1, dic2):
    """
    Cosine similatiry, the query dictionary should go first
    :param dic1:
    :param dic2:
    :return:
    """
    def dt_prdc(d1, d2):
        suma = 0
        for x in d1:
            if x in d2.keys():
                suma += d1[x] * d2[x]
        return suma

    def clc(d):
        suma = 0
        for x in d:
            suma += d[x] * d[x]
        return math.sqrt(suma)

    # z = {}
    # for x in dic1:
    #     if x in dic2.keys():
    #         z[x] = dic2[x]

    tmp = (clc(dic1) * clc(dic2))
    if not tmp:
        return 0
    return dt_prdc(dic1, dic2) / tmp