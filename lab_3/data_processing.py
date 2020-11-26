import xml.etree.ElementTree as ET
import string
import pandas as pd
import re
import numpy as np



def get_file_root(pth=''):
    root = ET.parse(pth).getroot()
    return root

def parse_file(path_file):
    root = get_file_root(path_file)

    words, subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, vowels, befores, afters, lengths, \
            before_pauses, after_pauses, graphemes, phonemes, allophones, before_vowels, after_vowels = [], [], [], [], \
                                                                        [], [], [], [], [], [], [], [], [], [], [], [], []

    for item in root.findall('./sentence'):
        num_word = 0
        amount = 0

        for idx, child in enumerate(item):
            if child.tag == 'word':
                amount += 1
        for idx, child in enumerate(item):
            [word, subpart_of_speech, genesys, semantics1, semantics2, form, vowel, before, after, length,
            before_pause, after_pause] = [None]*12
            if child.tag == 'word':
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
                    after = amount - num_word

                    # вместо количества гласных и согласных в слове использовал длину слова и кол-во гласных
                    # (это равносильная замена)
                    length = len(word)

                # Морфограмматическая информация о слове (часть речи,
                # род, форма, одушевленность и т.п.)
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


                before_pause, after_pause = 0, 0
                if item[idx - 2].tag == 'pause':
                    before_pause = 1

                if item[idx + 2].tag == 'pause':
                    after_pause = 1


                vowel = 0
                constants_vowels = []
                for i in child.findall('./letter'):
                    if 'flag' in i.attrib:
                        if i.attrib['flag'] in ['4','8','9']:
                            vowel += 1
                            constants_vowels.append(1)
                        else:
                            constants_vowels.append(0)
                    else:
                        constants_vowels.append(0)


                for idx, letter, phoneme, allophone in zip(range(len(constants_vowels)), child.findall('./letter'), child.findall('./phoneme'), child.findall('./allophone')):
                    [f_grapheme, f_phoneme, f_allophone] = [None] * 3
                    if 'char' in letter.attrib:
                        f_grapheme = letter.attrib['char']
                    if 'ph' in phoneme.attrib:
                        f_phoneme = phoneme.attrib['ph']
                    if 'ph' in allophone.attrib:
                        f_allophone = allophone.attrib['ph']

                    if idx>0:
                        before_vowel = constants_vowels[idx-1]
                    else:
                        before_vowel = None
                    if idx<len(constants_vowels)-1:
                        after_vowel = constants_vowels[idx+1]
                    else:
                        after_vowel = None

                    before_vowels.append(before_vowel)
                    after_vowels.append(after_vowel)
                    graphemes.append(f_grapheme)
                    phonemes.append(f_phoneme)
                    allophones.append((f_allophone))
                    before_pauses.append(before_pause)
                    after_pauses.append(after_pause)
                    words.append(word)
                    afters.append(after)
                    befores.append(before)
                    lengths.append(length)
                    forms.append(form)
                    genesyss.append(genesys)
                    subpart_of_speechs.append(subpart_of_speech)
                    semantics2s.append(semantics2)
                    semantics1s.append(semantics1)
                    vowels.append(vowel)







    df = pd.DataFrame(data = [words, subpart_of_speechs, genesyss, semantics1s, semantics2s, forms, vowels, befores, afters, lengths,
            before_pauses, after_pauses, graphemes, phonemes, allophones, before_vowels, after_vowels])

    df = df.transpose()
    df.columns = ['word', 'subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'vowel', 'before', 'after', 'length',
                  'before_pause', 'after_pause', 'grapheme', 'phoneme', 'allophone', 'before_vowel', 'after_vowel']

    df.to_csv(r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features3.csv', index=False)




if __name__ == "__main__":
    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\gieroi_nashiegho_vriemieni.Result.xml'
    parse_file(path_file)
