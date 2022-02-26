



from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import math
import sys
import re
from collections import defaultdict
import gzip
import json
import warnings
import nltk
from nltk.tokenize import word_tokenize

from nltk import tokenize

import gensim

from gensim.test.utils import datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer


def getWordmap(wordVecPath):
    words=[]
    We = []
    f = open(wordVecPath,'r')
    lines = f.readlines()

    for (n,line) in enumerate(lines):
        if (n == 0) :
            print(line)
            continue
        word, vect = line.rstrip().split(' ', 1)
                    
        vect = np.fromstring(vect, sep=' ')
        We.append(vect)
        words.append(word)

        #if(n==200000):
        #    break
    f.close()       
    return (words, We)

def substitution_generation(sentence, prefix, source_word, glove_model,ps, lemmatizer, num_selection=50):

    nltk_sent = nltk.word_tokenize(sentence)
    words_tag = nltk.pos_tag(nltk_sent)
    nltk_prefix = nltk.word_tokenize(prefix)
    source_index = len(nltk_prefix)
    source_tag = words_tag[source_index][1]

    #print(source_tag)
    
    pre_tokens = []
    try:
        pre_tokens = glove_model.most_similar(positive=[source_word], topn=50)
    except KeyError:
        pre_tokens = []

    source_stem = ps.stem(source_word)
    source_lemm = lemmatizer.lemmatize(source_word)

    cur_tokens = []

    for i in range(len(pre_tokens)):
        token = pre_tokens[i][0]
        if(token==source_word):
            continue

        token_stem = ps.stem(token)
        if(token_stem == source_stem):
            continue

        token_lemm = lemmatizer.lemmatize(token)
        if source_lemm==token_lemm:
            continue

        nltk_sent[source_index] = token
        sub_sentence = " ".join(word for word in nltk_sent)

        
        nltk_sub_sent = nltk.word_tokenize(sub_sentence)
        sub_words_tag = nltk.pos_tag(nltk_sub_sent)
        sub_tag = sub_words_tag[source_index][1]

        #print(sub_sentence)

        #print(token,sub_words_tag[source_index],sub_tag)

        if sub_tag != source_tag:
            continue

        cur_tokens.append(token)
       
        if(len(cur_tokens)==num_selection):
            break

    return cur_tokens

def find_sentence(sentence, original_word, offset):


  target_allindex = [i for i in range(len(sentence)) if sentence.startswith(original_word, i)]

  #print(target_allindex)

  target_index = -1
  if offset in target_allindex:
    target_index = target_allindex.index(offset)
    #print(target_index)

  if target_index == -1:
    print("target is not found!!!!")

  #print("target_index: ", target_index)

  clean_sent  = " ".join(sentence.split())

  
  clean_sent = clean_sent.replace("’","'")
  clean_sent = clean_sent.replace("“","'")
  clean_sent = clean_sent.replace("”","'")
  #print(clean_sent)
  sub_sents = tokenize.sent_tokenize(clean_sent)
  target_count = -1
  old_index = -1
  current_sent = ""
  prefix = ""
  is_found = False

  last_sent = ""

  for sub in sub_sents:

    #sub = sub.replace("’s","'s")

    #print("sub:", sub)

    sub_allindex = [i for i in range(len(sub)) if sub.startswith(original_word, i)]

    for index in sub_allindex:
      target_count += 1
      if target_count == target_index:

        current_sent = sub

        prefix = sub[:index]

        if len(current_sent) < 40:
          if last_sent != "":
            current_sent = last_sent + " " + current_sent
            prefix = last_sent + " " + prefix
        is_found = True
        break
      last_sent = sub

    if is_found:
      break
  return current_sent,prefix

def main():
    
    eval_dir = "data/swords-v1.1_test.json"
    with open(eval_dir, 'r') as f:
      swords = json.load(f)

    print("loading embeddings ...")

    wordVecPath = "/media/nlp/ff4212ed-997e-4bf5-bd3a-5d9b5a68b68f/2T/glove.6B.300d.txt"


    glove_file = datapath(wordVecPath)

    tmp_file = get_tmpfile('glove_word2vec.txt')

    glove2word2vec(glove_file,tmp_file)

    glove_model = KeyedVectors.load_word2vec_format(tmp_file)

    ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer()
    
    output_SR_file= "/home/nlp/Desktop/swords/swords-v1.1_test_Embeddings.json"

    tid_to_sids = defaultdict(list)
    for sid, substitute in swords['substitutes'].items():
      tid_to_sids[substitute['target_id']].append(sid)

    # NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "run") or False if your method produces wordforms (e.g. "ran")
    result = {'substitutes_lemmatized': True, 'substitutes': {}}
    errors = 0

    i = 0

    for tid, target in swords['targets'].items():

      i += 1
      #if i <= 716:
      #  continue

      #print('Sentence {} rankings: '.format(i))

      context = swords['contexts'][target['context_id']]
      substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
      labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
      scores = [l.count('TRUE') / len(l) for l in labels]

      target_word = target['target']

      print(target_word)

      substitutes = []


      #substitutes = ppdb_model.predict(target_word)
      current_sent,prefix = find_sentence(context['context'],target['target'],target['offset'])

      substitutes = substitution_generation(current_sent, prefix, target_word, glove_model,ps,lemmatizer)

      print(substitutes)

      scores = [random.random() for _ in substitutes]

      result['substitutes'][tid] = list(zip(substitutes, scores))

      #break
      #if i==30:
      #  break

    with open(output_SR_file, 'w') as f:
      f.write(json.dumps(result))


if __name__ == "__main__":
    main()

