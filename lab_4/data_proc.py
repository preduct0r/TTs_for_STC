import xml.etree.ElementTree as ET
import string
import pandas as pd
import re
import numpy as np
from sklearn import preprocessing



def get_file_root(pth=''):
    root = ET.parse(pth).getroot()
    return root

def parse_file(path_file):
    root = get_file_root(path_file)

    words, befores, afters, pauses, pause_types, phonemes, phonemes_word, phonemes_sent, allophones, otns, ens, mfccs = \
        [], [], [], [], [], [], [], [], [], [], [], []

    for item in root.findall('./sentence'):
        num_word = 0
        phoneme_sent = 0
        word_amount = 0
        phoneme_amount = 0


        for idx, child in enumerate(item):
            if child.tag == 'word':
                word_amount += 1
                for phoneme in child.findall('./phoneme'):
                    phoneme_amount += 1


        for idx, child in enumerate(item):
            [word, before, after] = [None] * 3
            if child.tag == 'word':
                phoneme_word = 0
                num_word += 1

                if 'original' in child.attrib:
                    word = child.attrib['original']
                    while not (len(word)>0 or word[-1].isalpha()):
                        word = word[:-1]
                    while not (len(word)>0 or word[0].isalpha()):
                        word = word[1:]

                    # вместо количества слов в предложении и позиции слова использовал кол-во слов после слова и до
                    # (это равносильная замена)
                    before = num_word - 1
                    after = word_amount - num_word


                for phoneme, allophone in zip(child.findall('./phoneme'), child.findall('./allophone')):
                    [f_phoneme, f_allophone, otn, en, mfcc] = [None] * 5
                    phoneme_sent += 1
                    phoneme_word += 1

                    if 'ph' in phoneme.attrib:
                        f_phoneme = phoneme.attrib['ph']
                    if 'ph' in allophone.attrib:
                        f_allophone = allophone.attrib['ph']
                    if 'OtN' in allophone.attrib:
                        otn = allophone.attrib['OtN']
                    if 'En' in allophone.attrib:
                        en = list(allophone.attrib['En'][1:-1].split('|'))
                    if 'mfcc' in allophone.attrib:
                        mfcc = list(allophone.attrib['mfcc'][1:-1].split('|'))


                    phonemes.append(f_phoneme)
                    allophones.append((f_allophone))
                    words.append(word)
                    afters.append(after)
                    befores.append(before)
                    otns.append(otn)
                    ens.append(en)
                    phonemes_sent.append(phoneme_sent)
                    phonemes_word.append(phoneme_word)
                    mfccs.append(mfcc)


            [pause, pause_type] = [None] * 2
            if child.tag == 'pause':
                pause = 1
                pause_type = child.attrib['type']

            pauses.append(pause)
            pause_types.append(pause_type)






    df = pd.DataFrame(data = [words, befores, afters, pauses, pause_types, phonemes, phonemes_word,
                              phonemes_sent, allophones, otns, ens, mfccs])

    df = df.transpose()
    df.columns = ['word', 'before', 'after', 'pause', 'pause_type', 'phoneme', 'phoneme_word',
                  'phoneme_sent', 'allophone', 'otn', 'en', 'mfcc']

    df = df.fillna(-1)
    to_convert = ['pause', 'pause_type', 'word', 'phoneme', 'allophone', 'otn']
    for col in to_convert:
        df[col] = df[col].astype('category')

    min_max_scaler = preprocessing.MinMaxScaler()
    df.loc[:, ['before', 'after', 'phoneme_word', 'phoneme_sent']] = min_max_scaler.fit_transform(
        df.loc[:, ['before', 'after', 'phoneme_word', 'phoneme_sent']])


    df.to_csv(r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features_lab4.csv', index = False)




if __name__ == "__main__":
    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\gieroi_nashiegho_vriemieni.Result.xml'
    parse_file(path_file)
