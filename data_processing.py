# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import xml.etree.ElementTree as ET
import string
import csv
import pandas as pd
import re
import numpy as np
import ast


def get_file_root(pth=''):


    root = ET.parse(pth).getroot()
    return root

def parse_file():
    root = get_file_root('data/veshnie-vody.Result.xml')
    corpus = []
    vectors = []
    punctuation_regex = re.compile("[.?!;]")

    for item in root.findall('./sentence'):
        for idx, child in enumerate(item):
            if child.tag == 'word' and 'original' in child.attrib:
                vector_i = {}
                corpus_i = {}
                word = child.attrib['original']
                rm_i = []
                en_i = []
                features_i = []
                mfcc_i = []
                if len(child.findall("./allophone"))>1:
                    for letter in child.findall("./allophone"):
                        letter_i = letter.attrib
                        rm_i.append(letter_i["Rm"])
                        en_i.append(letter_i["En"])
                        features_i.append(letter_i["features"])
                        mfcc_i.append(letter_i["mfcc"])

                    vector_i['word'] = word[:-1] if len(punctuation_regex.findall(word[:-1]))>0 else word
                    vector_i['Rm'] = rm_i
                    vector_i['En'] = en_i
                    vector_i["features"] = features_i
                    vector_i["mfcc"] = mfcc_i

                    corpus_i["word"] = word[:-1] if len(punctuation_regex.findall(word[:-1]))>0 else word

                    if len(punctuation_regex.findall(word))>0:
                        ispunct = 1
                        punct = word[-1]
                    else:
                        ispunct = 0
                        punct = 'a'

                    vector_i['ispunctuation'] = ispunct
                    vector_i['punct'] = punct
                    corpus_i['ispunctuation'] = ispunct
                    corpus_i['punct'] = punct
                    corpus.append(corpus_i)
                    vectors.append(vector_i)
    return corpus, vectors

def savetoCSV(newsitems, filename, fields):
    # specifying the fields for csv file

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(newsitems)
    # Press the green button in the gutter to run the script.
def convert_feats_to_tensor(arr, pth):
    rm = arr
    rm = [ast.literal_eval(rm_i) for rm_i in rm]
    rms_padded = []
    max_rm = max([len(l) for l in rm])
    for i in rm:
        pad_len = max_rm - len(i)
        padded = [int(ii) for ii in i] + [0] * pad_len
        rms_padded.append(padded)
    np.save(pth, np.array(rms_padded))

def process():
    corpus, vectors = parse_file()
    # savetoCSV(corpus, './data/corpus.csv', list(corpus[0].keys()))
    # savetoCSV(vectors, './data/vectors.csv', list(vectors[0].keys()))

    df2 = pd.read_csv('./data/vectors.csv', encoding='latin-1')
    features = []
    lens = []
    for row in df2["features"].values:
        l = [[re.sub('\D', '', i) for i in feat[1:-1].split("|") if re.sub('\D', '', i)!=''] for feat in row.split(", ")]
        lens.append(len(l))
        features.append(l)
    features = np.array(features)
    # np.save("./features.npy", features)
    vectors = pd.read_csv('./data/vectors.csv', encoding='latin-1')
    convert_feats_to_tensor(vectors["Rm"], "./rms.npy")

#     print(min(lens), max(lens), np.mean(lens))

