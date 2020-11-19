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

def parse_file(path_file):
    root = get_file_root(path_file)

    punctuation_regex = re.compile("[.?!;]")
    words, subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, reducts, upper_cases, vowels, eees, befores, afters, lengths, \
            PunktBegs, PunktEnds, EmphBegs, EmphEnds = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for item in root.findall('./sentence'):
        amount = 0
        for idx, child in enumerate(item):
            if child.tag == 'word':
                amount += 1
        for idx, child in enumerate(item):
            num_word = 0
            [word, subpart_of_speech, genesys, semantics1, semantics2, form,
                     reduct, upper_case, vowel, eee, before, after, length] = [None]*13
            if child.tag == 'word':
                num_word += 1

                if 'original' in child.attrib:
                    word = child.attrib['original']
                    before = num_word = 1
                    befores.append(before)
                    after = amount - num_word
                    afters.append(after)
                    length = len(word)
                    lengths.append(length)
                for i in child.findall('./dictitem'):
                    if 'subpart_of_speech' in i.attrib:
                        subpart_of_speech = i.attrib['subpart_of_speech']
                        subpart_of_speechs.append(subpart_of_speech)
                    elif 'genesys' in i.attrib:
                        genesys = i.attrib['genesys']
                        genesyss.append(genesys)
                    elif 'semantics1' in i.attrib:
                        semantics1 = i.attrib['semantics1']
                        semantics1s.append(semantics1)
                    elif 'semantics2' in i.attrib:
                        semantics2 = i.attrib['semantics2']
                        semantics2s.append(semantics2)
                    elif 'form' in i.attrib:
                        form = i.attrib['form']
                        forms.append(form)
                for i in child.findall('./letter'):
                    if 'reduct' in i.attrib:
                        reduct = i.attrib['reduct']
                        reducts.append(reduct)
                    if 'flag' in i.attrib:
                        if i.attrib['flag'] ==10:
                            upper_case = 1
                        else:
                            upper_case = 0
                        upper_cases.append(upper_case)
                        if i.attrib['flag'] == 8:
                            vowel = 1
                        else:
                            vowel = 0
                        vowels.append(vowel)
                        if i.attrib['flag'] == 4:
                            eee = 1
                        else:
                            eee = 0
                        eees.append(eee)
            [PunktBeg, PunktEnd, EmphBeg, EmphEnd] = [None]*4
            if child.tag == 'content':
                if 'PunktBeg' in child.attrib:
                    PunktBeg = child.attrib['PunktBeg']
                elif 'PunktEnd' in child.attrib:
                    PunktEnd = child.attrib['PunktEnd']
                elif 'EmphBeg' in child.attrib:
                    EmphBeg = child.attrib['EmphBeg']
                elif 'EmphEnd' in child.attrib:
                    EmphBeg = child.attrib['EmphEnd']

    df = pd.DataFrame(columns= ['word', 'subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'reduct', 'upper_case',
                                'vowel', 'eee', 'before', 'after', 'length', 'PunktBeg', 'PunktEnd', 'EmphBeg', 'EmphEnd'], data = [words, \
                      subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, reducts, upper_cases, vowels, eees, befores, afters, lengths, \
            PunktBegs, PunktEnds, EmphBegs, EmphEnds])

    df.to_csv('./data/features.csv', index=False)




#                 features_i = []
#                 mfcc_i = []
#                 if len(child.findall("./allophone"))>1:
#                     for letter in child.findall("./allophone"):
#                         letter_i = letter.attrib
#                         rm_i.append(letter_i["Rm"])
#                         en_i.append(letter_i["En"])
#                         features_i.append(letter_i["features"])
#                         mfcc_i.append(letter_i["mfcc"])
#
#                     vector_i['word'] = word[:-1] if len(punctuation_regex.findall(word[:-1]))>0 else word
#                     vector_i['Rm'] = rm_i
#                     vector_i['En'] = en_i
#                     vector_i["features"] = features_i
#                     vector_i["mfcc"] = mfcc_i
#
#                     corpus_i["word"] = word[:-1] if len(punctuation_regex.findall(word[:-1]))>0 else word
#
#                     if len(punctuation_regex.findall(word))>0:
#                         ispunct = 1
#                         punct = word[-1]
#                     else:
#                         ispunct = 0
#                         punct = 'a'
#
#                     vector_i['ispunctuation'] = ispunct
#                     vector_i['punct'] = punct
#                     corpus_i['ispunctuation'] = ispunct
#                     corpus_i['punct'] = punct
#                     corpus.append(corpus_i)
#                     vectors.append(vector_i)
#     return corpus, vectors
#
# def savetoCSV(newsitems, filename, fields):
#     # specifying the fields for csv file
#
#     # writing to csv file
#     with open(filename, 'w') as csvfile:
#         # creating a csv dict writer object
#         writer = csv.DictWriter(csvfile, fieldnames=fields)
#
#         # writing headers (field names)
#         writer.writeheader()
#
#         # writing data rows
#         writer.writerows(newsitems)
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

def process(path_file):
    corpus, vectors = parse_file(path_file)
    savetoCSV(corpus, './data/corpus.csv', list(corpus[0].keys()))
    savetoCSV(vectors, './data/vectors.csv', list(vectors[0].keys()))

    df2 = pd.read_csv('./data/vectors.csv', encoding='latin-1')
    features = []
    lens = []
    for row in df2["features"].values:
        l = [[re.sub('\D', '', i) for i in feat[1:-1].split("|") if re.sub('\D', '', i)!=''] for feat in row.split(", ")]
        lens.append(len(l))
        features.append(l)
    features = np.array(features)
    np.save("./features.npy", features)
    vectors = pd.read_csv('./data/vectors.csv', encoding='latin-1')
    convert_feats_to_tensor(vectors["Rm"], "./rms.npy")

    print(min(lens), max(lens), np.mean(lens))

if __name__ == "__main__":
    path_file = 'data/veshnie-vody.Result.xml'
    process(path_file)
