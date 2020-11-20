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

    punctuation_regex = re.compile("[\"\'.-?!;]")

    words, subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, reducts, upper_cases, vowels, befores, afters, lengths, \
            PunktBegs, PunktEnds, EmphBegs, EmphEnds, intonations, pauses = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for item in root.findall('./sentence'):
        num_word = 0

        amount = 0
        for idx, child in enumerate(item):
            if child.tag == 'word':
                amount += 1
        for idx, child in enumerate(item):
            [word, subpart_of_speech, genesys, semantics1, semantics2, form,
                     reduct, upper_case, vowel, before, after, length] = [None]*12
            if child.tag == 'word':
                num_word += 1

                if 'original' in child.attrib:
                    word = child.attrib['original']
                    while not (len(word)>0 or word[-1].isalpha()):
                        word = word[:-1]
                    while not (len(word)>0 or word[0].isalpha()):
                        word = word[1:]

                    before = num_word - 1
                    after = amount - num_word
                    length = len(word)
                for i in child.findall('./dictitem'):
                    if 'subpart_of_speech' in i.attrib:
                        subpart_of_speech = i.attrib['subpart_of_speech']
                    if 'genesys' in i.attrib:
                        genesys = i.attrib['genesys']
                    if 'semantics1' in i.attrib:
                        semantics1 = i.attrib['semantics1']
                    if 'semantics2' in i.attrib:
                        semantics2 = i.attrib['semantics2']
                    if 'form' in i.attrib:
                        form = i.attrib['form']
                    break
                reduct, upper_case, vowel = [], 0, []
                for i in child.findall('./letter'):
                    if 'reduct' in i.attrib:
                        reduct.append(i.attrib['reduct'])
                    if 'flag' in i.attrib:
                        if i.attrib['flag'] ==10:
                            upper_case = 1
                        if i.attrib['flag'] == 8:
                            v = 1
                        else:
                            v = 0
                        vowel.append(v)

                words.append(word)
                afters.append(after)
                befores.append(before)
                lengths.append(length)
                forms.append(form)
                genesyss.append(genesys)
                subpart_of_speechs.append(subpart_of_speech)
                semantics2s.append(semantics2)
                semantics1s.append(semantics1)
                reducts.append(reduct)
                upper_cases.append(upper_case)
                vowels.append(vowel)


                [PunktBeg, PunktEnd, EmphBeg, EmphEnd] = [None]*4
                if item[idx-1].tag == 'content':
                    child = item[idx-1]
                    if 'PunktBeg' in child.attrib:
                        PunktBeg = child.attrib['PunktBeg']
                    elif 'PunktEnd' in child.attrib:
                        PunktEnd = child.attrib['PunktEnd']
                    elif 'EmphBeg' in child.attrib:
                        EmphBeg = child.attrib['EmphBeg']
                    elif 'EmphEnd' in child.attrib:
                        EmphEnd = child.attrib['EmphEnd']
                PunktBegs.append(PunktBeg)
                PunktEnds.append(PunktEnd)
                EmphBegs.append(EmphBeg)
                EmphEnds.append(EmphEnd)

                [intonation, pause] = [None] * 2
                if item[idx + 1].tag == 'intonation':
                    child = item[idx + 1]
                    if 'type' in child.attrib:
                        intonation = child.attrib['type']

                intonations.append(intonation)

                if item[idx + 2].tag == 'pause':
                    child = item[idx + 2]
                    if 'type' in child.attrib:
                        pause = child.attrib['type']

                pauses.append(pause)





    df = pd.DataFrame(data = [words,
                      subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, reducts, upper_cases, vowels, befores, afters, lengths, \
            PunktBegs, PunktEnds, EmphBegs, EmphEnds, intonations, pauses])

    df = df.transpose()
    df.columns = ['word', 'subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'reduct', 'upper_case',
                                'vowel', 'before', 'after', 'length', 'PunktBeg', 'PunktEnd', 'EmphBeg', 'EmphEnd', 'intonation', 'pause']

    df.to_csv(r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features.csv', index=False)



    # Press the green button in the gutter to run the script.

def process(path_file):
    parse_file(path_file)


if __name__ == "__main__":
    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\gieroi_nashiegho_vriemieni.Result.xml'
    process(path_file)
