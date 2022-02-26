



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
from nltk import tokenize

from sklearn.metrics.pairwise import cosine_similarity as cosine

import numpy as np
import torch
import nltk

from pathlib import Path
import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from fairseq.models.transformer import TransformerModel


from transformers import AutoTokenizer, AutoModelWithLMHead

from spacy.lang.en import English

from bert_score.scorer import BERTScorer

import pdb


def containSuffix(suffix_words, sent_suffix, complex_word):

    if sent_suffix=="":
        for suffix in suffix_words:
            if suffix == "":
                return True 
    
    for suffix in suffix_words:
        if suffix == "":
            continue

        min_len = min(len(suffix),len(sent_suffix))
        for i in range(min_len):

            if i==0 or i==1:
              if suffix[i] == sent_suffix[i]:
                return True

            if len(suffix[i])>5 and len(sent_suffix[i])>5 and suffix[i][:5] == sent_suffix[i][:5]:
                return True

    return False

ps = PorterStemmer()

def containComplexInSuffix(complex_word, sent_suffix):

    for word in sent_suffix:
        if word == complex_word:
            return True

    return False

def extractPhrase(suffix_words, complex_word, rest_words):


    #pdb.set_trace()
    complex_stem = ps.stem(complex_word)


    for suffix in suffix_words:
        if(len(suffix) != 3 or complex_word in rest_words):
            continue

        str_suffix = str(suffix)

        is_continue = True

        for word in rest_words[:3]:
            if complex_stem == ps.stem(word):
                is_continue = False
                break

        if is_continue and str_suffix == str(rest_words[2:5]):
            phrase = " ".join(word for word in rest_words[:2])
            return phrase

        if is_continue and str_suffix == str(rest_words[3:6]):
            phrase = " ".join(word  for word in rest_words[:3])
            return phrase
    return ""

def extract_substitute(output_sentences, original_sentence, complex_word, prefix):


    prefix_words = nltk.word_tokenize(prefix)
    prefix_allindex = [i for i,x in enumerate(prefix_words) if x==complex_word]

    num = len(prefix_allindex)

    original_words = nltk.word_tokenize(original_sentence)

    len_original_words = len(original_words)

    indices = [i for i,x in enumerate(original_words) if x==complex_word]

    #print('num:', num, prefix_allindex)
    #print('indices:', indices)
    #pdb.set_trace()

    index_of_complex_word = -1

    if len(indices)>0:
      index_of_complex_word = indices[num]
    else:
      print("******************no found the complex word*****************")
      return [],[]

    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])
    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4

    
    suffix_words = []

    if index_of_complex_word+1 < len_original_words:
        suffix_words.append(original_words[index_of_complex_word+1]) #suffix_words.append(original_words[index_of_complex_word+1:min(index_of_complex_word+4,len_original_words)])
    else:
        suffix_words.append("")
    
    #pdb.set_trace()

    for sentence in output_sentences:

        if len(sentence)<3:
            continue

        words = nltk.word_tokenize(sentence)

        if index_of_complex_word>=len(words):
            continue

        if words[index_of_complex_word] == complex_word:
           
            len_words = len(words)
            if index_of_complex_word+1 < len_words:
                suffix = words[index_of_complex_word+1]#words[index_of_complex_word+1:min(index_of_complex_word+4,len_words)]

                if suffix not in suffix_words:
                    suffix_words.append(suffix)
            #else:
                
            #    if suffix_words[0] != "":
            #        suffix_words.append("")
            #break

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set()
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    
   
    #print("suffix_words: ", suffix_words)

    id_sent = 0

    substitutes = []
    scores = []

    #pdb.set_trace()

    for sentence in output_sentences:

        if len(sentence)<3:
            continue
        #print(sentence)
        id_sent += 1

        words = nltk.word_tokenize(sentence)

 
        if index_of_complex_word>=len(words):
            continue

        candi = words[index_of_complex_word].lower()

        candi_stem = ps.stem(candi)

        if candi_stem in not_candi or candi in not_candi:
            continue

        #candi_pos = nltk.pos_tag(words)[index_of_complex_word][1]

        #print(words, " ------ [", index_of_complex_word, "]", candi)
     
        len_words = len(words)

        sent_suffix = ""
        
        if index_of_complex_word + 1 < len_words:
            sent_suffix = words[index_of_complex_word+1]

        #sent_suffix = words[index_of_complex_word+1:min(index_of_complex_word+4,len_words)]
        

        #is_same_suffix = containSuffix(suffix_words, sent_suffix)


        #print(is_same_suffix)


        if sent_suffix in suffix_words:
            if candi not in substitutes:
                substitutes.append(candi)


    return substitutes, (context,index_of_complex_in_context)

def extract_substitute2(output_sentences, original_sentence, complex_word, prefix):


    prefix_words = nltk.word_tokenize(prefix)
    prefix_allindex = [i for i,x in enumerate(prefix_words) if x==complex_word]

    num = len(prefix_allindex)

    original_words = nltk.word_tokenize(original_sentence)

    len_original_words = len(original_words)

    indices = [i for i,x in enumerate(original_words) if x==complex_word]

    #print('num:', num, prefix_allindex)
    #print('indices:', indices)

    index_of_complex_word = -1

    if len(indices)>0:
      index_of_complex_word = indices[num]
    else:
      print("******************no found the complex word*****************")
      return []

    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4

    substitutes = []

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set(['the', 'with', 'of', 'a', 'an' , 'for' , 'in'])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    
    id_sent = 0
    for sentence in output_sentences:

        if len(sentence)<3:
            continue
        #print(sentence.lower())
        id_sent += 1

        words = nltk.word_tokenize(sentence)

        if index_of_complex_word>=len(words):
            continue

        candi = words[index_of_complex_word].lower()

        candi_stem = ps.stem(candi)

        if candi_stem in not_candi or candi in not_candi:
            continue

        if candi not in substitutes:
            #print(candi)
            substitutes.append(candi)

    return substitutes, (context,index_of_complex_in_context)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

def substitutes_BertScore(context, target, substitutes):

    refs = []
    cands = []

    target_id = context[1]
    sent = context[0]

    words = sent.split(" ")
    for sub in substitutes:
        refs.append(sent)
        
        new_sent = ""
        
        for i in range(len(words)):
            if i==target_id:
                new_sent += sub + " "
            else:
                new_sent += words[i] + " "
        cands.append(new_sent.strip())

    P, R, F1 = scorer.score(cands, refs)

    #print(cands)
    #print(F1)

    return F1
    
def find_substitutes_direct(model, sentence, target, prefix, beam):

    
    #print("sentence:", sentence)
    #print("prefix: ", prefix)
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    sentence_tokens = model.encode(sentence)

    target_tokens = model.encode(target)

    #print("prefix_len = ", len(prefix_tokens[0]))

    attn_len = len(prefix_tokens[0])+len(target_tokens)-1
    #attn_len = (len(prefix_tokens[0]), len(prefix_tokens[0])+len(target_tokens)-1)


    outputs, parascores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens)
    #outputs = model.generate(sentence_tokens, beam=20)
    #print(outputs)
    output_sentences = [model.decode(x['tokens']) for x in outputs]

    #first_substitute,frequency_substitute,substitutes = extract_substitute(output_sentences, sentence, prefix, complex_word, fasttext)
    #substitutes, context = extract_substitute2(output_sentences, sentence, target, prefix)
    for sent in output_sentences:
        print(sent)

    substitutes = extract_substitute(output_sentences, sentence, target, prefix)
   

    
    return substitutes

def find_substitutes(model, sentence, target, prefix, beam, bertscore):

    
    #print("sentence:", sentence)
    #print("prefix: ", prefix)
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    sentence_tokens = model.encode(sentence)

    target_tokens = model.encode(target)

    #print("prefix_len = ", len(prefix_tokens[0]))

    attn_len = len(prefix_tokens[0])+len(target_tokens)-1
    #attn_len = (len(prefix_tokens[0]), len(prefix_tokens[0])+len(target_tokens)-1)


    outputs, parascores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens, attn_len=attn_len)
    #outputs, parascores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens)
    #outputs = model.generate(sentence_tokens, beam=20)
    #print(outputs)
    output_sentences = [model.decode(x['tokens']) for x in outputs]

    

    #first_substitute,frequency_substitute,substitutes = extract_substitute(output_sentences, sentence, prefix, complex_word, fasttext)
    substitutes, context = extract_substitute(output_sentences, sentence, target, prefix)
    
    for sent in output_sentences:
        print(sent)

    #if len(substitutes) == 0:
    #    for sentence in output_sentences:
    #        print(sentence)

    #print('context:', context[0], ' the index of complex word in context:', context[1])

    #print(substitutes)
    filter_substitutes = []

    scores = []

    if len(substitutes)>0:
        bert_scores = substitutes_BertScore(context, target, substitutes)

        #print(bert_scores)
        #threshold = bertscore
        max_score = torch.max(bert_scores)

        threshold = max_score - 0.1 
        if threshold<bertscore:
            threshold = bertscore
        for i in range(len(substitutes)):
            if(bert_scores[i]>threshold):
                filter_substitutes.append(substitutes[i])
                scores.append(float(bert_scores[i]))
        
    #count_rank = [rank_bert.index(v)+1 for v in bert_scores]
    
    return substitutes, filter_substitutes, scores


def nlp_tokenize(sentence):
  # Load English tokenizer, tagger, parser, NER and word vectors

  nlp = English()
  my_doc = nlp(sentence)

  # Create list of word tokens

  token_list = []
  for token in my_doc:
    token_list.append(token.text)

  return token_list

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
 # clean_sent = clean_sent.replace("\"","'")
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
    parser = argparse.ArgumentParser()

    ## Required parameters
    
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data dir.")
    parser.add_argument("--paraphraser_path", default=None, type=str, required=True,
                        help=" the checkpoint path of paraphraser")
    parser.add_argument("--paraphraser_model", default=None, type=str, required=True,
                        help=" the checkpoint path of paraphraser")

    parser.add_argument("--bpe", default=None, type=str, required=True,
                        help=" which bpe")

    parser.add_argument("--bpe_codes", default=None, type=str, required=True,
                        help=" which bpe")

    parser.add_argument("--paraphraser_dict", default=None, type=str, required=True,
                        help=" the dict path of paraphraser")

    parser.add_argument("--output_SR_file",
                        default=None,
                        type=str,
                        help="The output directory of writing substitution selection.")
    parser.add_argument("--beam",
                        default=20,
                        type=int,
                        help="The number of beam.")
    parser.add_argument("--bertscore",
                        default=0.8,
                        type=float,
                        help="The value of bertscore.")
 
    args = parser.parse_args()


    en2en = TransformerModel.from_pretrained(args.paraphraser_path, checkpoint_file=args.paraphraser_model,
        data_name_or_path=args.paraphraser_dict, bpe=args.bpe, bpe_codes=args.bpe_codes)

    #print(args.eval_dir)
    
    with open(args.eval_dir, 'r') as f:
      swords = json.load(f)

    tid_to_sids = defaultdict(list)
    for sid, substitute in swords['substitutes'].items():
      tid_to_sids[substitute['target_id']].append(sid)

    # NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "run") or False if your method produces wordforms (e.g. "ran")
    result = {'substitutes_lemmatized': True, 'substitutes': {}}
    errors = 0

    i = 0

    print("beam:", args.beam, " bertscore:", args.bertscore)

    for tid, target in swords['targets'].items():

      i += 1
      #if i < 64:
      #  continue

      print('Sentence {} rankings: '.format(i))

      context = swords['contexts'][target['context_id']]
      substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
      labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
      scores = [l.count('TRUE') / len(l) for l in labels]

      #print(context['context'])

      current_sent,prefix = find_sentence(context['context'],target['target'],target['offset'])
      print(current_sent)

      #print(prefix)
      print(target['target'])
      print(', '.join(['{} ({}%)'.format(substitute['substitute'], round(score * 100)) for substitute, score in sorted(zip(substitutes, scores), key=lambda x: -x[1])]))
      #print(labels)
      #print(', '.join(['{} ({}%)'.format(substitute['substitute'], round(score * 100)) for substitute, score in sorted(zip(substitutes, scores), key=lambda x: -x[1])]))

      #print('-' * 80)
      substitutes, filter_substitutes, scores = find_substitutes(en2en,current_sent,target['target'],prefix,args.beam, args.bertscore)
      #substitutes = find_substitutes_direct(en2en,current_sent,target['target'],prefix,args.beam)

      print(substitutes)
      print(filter_substitutes)

      #scores = [random.random() for _ in substitutes]

      result['substitutes'][tid] = list(zip(filter_substitutes, scores))

      #print(substitutes)
      print('-' * 80)

      #break
      if i==16:
        break

    with open(args.output_SR_file, 'w') as f:
      f.write(json.dumps(result))


if __name__ == "__main__":
    main()

