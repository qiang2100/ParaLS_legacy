#!/usr/bin/python
# -*- coding: UTF-8 -*-



from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import math
import sys
import re

from sklearn.metrics.pairwise import cosine_similarity as cosine

import numpy as np
import torch
import nltk

import pdb

from pathlib import Path
import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from fairseq.models.transformer import TransformerModel

from transformers import AutoTokenizer, AutoModelWithLMHead

from nltk.stem import PorterStemmer

from bert_score.scorer import BERTScorer


ps = PorterStemmer()

scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def containSuffix(suffix_words, sent_suffix, complex_word):

    if sent_suffix=="":
        for suffix in suffix_words:
            if suffix == "":
                return True 
    
    for suffix in suffix_words:
        if suffix == sent_suffix:
            return True

        
    return False

def containComplexInSuffix(complex_word, sent_suffix):

    for word in sent_suffix:
        if word == complex_word:
            return True

    return False

def extractSubstitute_suffix(output_sentences, original_sentence, complex_word):

    len_suffix = 2

    original_words = nltk.word_tokenize(original_sentence)

    index_of_complex_word = -1

    if complex_word  not in original_words:
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)
    
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return []

    suffix_words = []

    len_original_words = len(original_words)

    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4
    
    if index_of_complex_word+1 < len_original_words:

        sss = original_words[index_of_complex_word+1:min(index_of_complex_word+len_suffix,len_original_words)]

        sss = " ".join(word for word in sss)

        suffix_words.append(sss.strip())
    else:
        suffix_words.append("")

    first_substitute = ""
    frequency_substitute = ""
    substitutes = []

    is_find_complex_word = False

    if output_sentences[0].find('<unk>'):

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    for sentence in output_sentences:

        if len(sentence)<3:
            continue
        words = nltk.word_tokenize(sentence)

        if index_of_complex_word>=len(words):
            continue

        if words[index_of_complex_word] == complex_word:
            is_find_complex_word = True 
            len_words = len(words)
            if index_of_complex_word+1 < len_words:
                suffix = words[index_of_complex_word+1:min(index_of_complex_word+len_suffix,len_words)]

                sss = " ".join(word for word in suffix)


                if sss.strip() != suffix_words[0]:
                    suffix_words.append(sss.strip())
            else:
                
                if suffix_words[0] != "":
                    suffix_words.append("")
            break
    #if is_find_complex_word == False:
    #    for sentence in output_sentences:
    #        print(sentence)
    #    return complex_word, complex_word, substitutes

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set(['the', 'with', 'of', 'a', 'an' , 'for' , 'in'])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    
    for suffix in suffix_words:
        if suffix !="":
            not_candi.add(suffix[0])

    #print("suffix_words: ", suffix_words)

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

        #candi_pos = nltk.pos_tag(words)[index_of_complex_word][1]

        #print(words, " ------ [", index_of_complex_word, "]", candi)
     
        len_words = len(words)
        
        sent_suffix = words[index_of_complex_word+1:min(index_of_complex_word+len_suffix,len_words)]

        sss = " ".join(word for word in sent_suffix)

        sent_suffix = sss.strip()


        
        if containComplexInSuffix(complex_word, sent_suffix):
            continue

        is_same_suffix = containSuffix(suffix_words, sent_suffix, complex_word)

        #print(is_same_suffix)

        #if is_same_suffix == False and id_sent < 4 and len_words>=index_of_complex_word+6:
        #    phrase = extractPhrase(suffix_words, complex_word, words[index_of_complex_word:index_of_complex_word+6])
        #    if phrase != "":
        #        substitutes.append(phrase)
        #        continue
        if is_same_suffix == False:
            continue

        if candi not in substitutes:
            #print(candi)
            substitutes.append(candi)
        #if(len(substitutes)==10):
        #    break

    #print(substitutes)

    if complex_word in substitutes:
        substitutes.remove(complex_word)

    return substitutes 

def read_eval_index_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            
            if not line:
                break
            
            sentence,words = line.strip().split('\t',1)
                #print(sentence)
            mask_word,labels = words.strip().split('\t',1)
            label = labels.split('\t')
                
            sentences.append(sentence)
            mask_words.append(mask_word)
                
            one_labels = []
            for la in label[1:]:
                la = la.strip()
                if la not in one_labels:
                    la_id,la_word = la.split(':')
                    one_labels.append(la_word)
                
                #print(mask_word, " ---",one_labels)
            mask_labels.append(one_labels)
            
    return sentences,mask_words,mask_labels

def read_eval_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id==1:
                    continue
                if not line:
                    break
                sentence,words = line.strip().split('\t',1)
                #print(sentence)
                mask_word,labels = words.strip().split('\t',1)
                label = labels.split('\t')
                
                sentences.append(sentence)
                mask_words.append(mask_word)
                
                one_labels = []
                for la in label:
                    la = la.strip()
                    if la not in one_labels:
                        one_labels.append(la)
                
                #print(mask_word, " ---",one_labels)
                    
                mask_labels.append(one_labels)
            else:
                if not line:
                    break
                #print(line)
                sentence,mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)
    return sentences,mask_words,mask_labels

def evaulation_SS_scores(ss,labels):
    assert len(ss)==len(labels)

    potential = 0
    instances = len(ss)
    precision = 0
    precision_all = 0.000001
    recall = 0
    recall_all = 0.000001

    for i in range(len(ss)):

        one_prec = 0
        
        common = list(set(ss[i]).intersection(labels[i]))

        if len(common)>=1:
            potential +=1
        precision += len(common)
        recall += len(common)
        precision_all += len(ss[i])
        recall_all += len(labels[i])

    potential /=  instances
    precision /= precision_all
    recall /= recall_all
    F_score = 2*precision*recall/(precision+recall)

    return potential,precision,recall,F_score

def extract_substitute(output_sentences, original_sentence, complex_word, threshold):

    original_words = nltk.word_tokenize(original_sentence)

    index_of_complex_word = -1

    if complex_word  not in original_words:
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)
    
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [],[]

    
    len_original_words = len(original_words)
    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4


    context = (context,index_of_complex_in_context)

    if output_sentences[0].find('<unk>'):

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set(['the', 'with', 'of', 'a', 'an' , 'for' , 'in'])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    

    #para_scores = []
    substitutes = []

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

    for sentence in output_sentences:

        if len(sentence)<3:
            continue
      
        words = nltk.word_tokenize(sentence)
        if index_of_complex_word>=len(words):
            continue
        candi = words[index_of_complex_word].lower()
        candi_stem = ps.stem(candi)
        if candi_stem in not_candi or candi in not_candi:
            continue

        len_words = len(words)
        sent_suffix = ""
        if index_of_complex_word + 1 < len_words:
            sent_suffix = words[index_of_complex_word+1]

        if sent_suffix in suffix_words:
            if candi not in substitutes:
                substitutes.append(candi)
   
    if len(substitutes)>0:
        bert_scores = substitutes_BertScore(context, complex_word, substitutes)

        #print(substitutes)
        bert_scores = bert_scores.tolist()

        #pdb.set_trace()


        filter_substitutes, bert_scores = filterSubstitute(substitutes, bert_scores, threshold)

        rank_bert = sorted(bert_scores,reverse = True )

        rank_bert_substitutes = [filter_substitutes[bert_scores.index(v)] for v in rank_bert]

        return filter_substitutes, rank_bert_substitutes

    return [],[]

def extractSubstitute_bertscore(output_sentences, original_sentence, complex_word, threshold):

    original_words = nltk.word_tokenize(original_sentence)

    index_of_complex_word = -1

    if complex_word  not in original_words:
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)
    
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [],[]

    
    len_original_words = len(original_words)
    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4


    context = (context,index_of_complex_in_context)

    if output_sentences[0].find('<unk>'):

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set(['the', 'with', 'of', 'a', 'an' , 'for' , 'in'])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    

    #para_scores = []
    substitutes = []

    for i in range(len(output_sentences)):

        sentence = output_sentences[i]

        if len(sentence)<3:
            continue

        words = nltk.word_tokenize(sentence)

        if index_of_complex_word>=len(words):
            continue

        candi = words[index_of_complex_word].lower()
        candi_stem = ps.stem(candi)
        if (candi_stem in not_candi) or (candi in not_candi):
            continue

        if candi not in substitutes:
            substitutes.append(candi)
            #para_scores.append(pre_scores[i])

    if len(substitutes)>0:
        bert_scores = substitutes_BertScore(context, complex_word, substitutes)

        #print(substitutes)
        bert_scores = bert_scores.tolist()

        #pdb.set_trace()


        filter_substitutes, bert_scores = filterSubstitute(substitutes, bert_scores, threshold)

        rank_bert = sorted(bert_scores,reverse = True )

        rank_bert_substitutes = [filter_substitutes[bert_scores.index(v)] for v in rank_bert]

        return filter_substitutes, rank_bert_substitutes

    return [],[]

   

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

    return F1

def filterSubstitute(substitutes, bert_scores, threshold):

    max_score = np.max(bert_scores)

    if max_score - 0.1 > threshold:
        threshold = max_score - 0.1

    filter_substitutes = []
    filter_bert_scores = []

    for i in range(len(substitutes)):
        if(bert_scores[i]>threshold):
            filter_substitutes.append(substitutes[i])
            filter_bert_scores.append(bert_scores[i])

    return filter_substitutes, filter_bert_scores


def lexicalSubstitute(model, sentence, complex_word, beam, threshold):

    index_complex = sentence.find(complex_word)

    prefix = ""

    if(index_complex != -1):
        prefix = sentence[0:index_complex]
        #print(prefix)
    else:
        #print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam, threshold)

    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    complex_tokens = model.encode(complex_word)

    sentence_tokens = model.encode(sentence)

    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1

    #outputs = model.generate2(sentence_tokens, beam=20, prefix_tokens=prefix_tokens)
    outputs,pre_scores = model.generate2(sentence_tokens, beam=beam, prefix_tokens=prefix_tokens, attn_len=attn_len)
    #print(outputs)
    
    output_sentences = [model.decode(x['tokens']) for x in outputs]

    bertscore_substitutes, ranking_bertscore_substitutes = extract_substitute(output_sentences, sentence, complex_word, threshold)

    #print(pre_scores)

    #for sen in output_sentences:
    #    print(sen)

    #bertscore_substitutes, ranking_bertscore_substitutes = extractSubstitute_bertscore(output_sentences, sentence, complex_word, threshold)
    #suffix_substitutes = extractSubstitute_suffix(output_sentences, sentence, complex_word)

    return bertscore_substitutes, ranking_bertscore_substitutes


   

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
                        required=True,
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

    output_sr_file = open(args.output_SR_file,"a+")
    output_sr_file.write(str("beam: "))
    output_sr_file.write('\t')
    output_sr_file.write(str(args.beam))
    output_sr_file.write('\t')
    output_sr_file.write(str("bertscore: "))
    output_sr_file.write('\t')
    output_sr_file.write(str(args.bertscore))
    output_sr_file.write('\n')

    en2en = TransformerModel.from_pretrained(args.paraphraser_path, checkpoint_file=args.paraphraser_model,
        data_name_or_path=args.paraphraser_dict, bpe=args.bpe, bpe_codes=args.bpe_codes)


    fileName = args.eval_dir.split('/')[-1][:-4]
    
    if fileName=='lex.mturk':
        eval_examples, complex_words, complex_labels = read_eval_dataset(args.eval_dir)
    else:
        eval_examples, complex_words, complex_labels = read_eval_index_dataset(args.eval_dir)

    eval_size = len(eval_examples)

    CS = []

    CS2 = []

    CS3 = []

    #output_sr_file.write("beam:", args.beam, " bertscore:", args.bertscore)
    #output_sr_file.write('\n')


    for i in range(eval_size):

        print("****************")
        print('Sentence {} rankings: '.format(i))

        print(eval_examples[i])

        print(complex_words[i])

        print(complex_labels[i])

            #if (i<498):
            #    continue
        #print("--------------------------")
        bert_substitutes, bert_rank_substitutes = lexicalSubstitute(en2en,eval_examples[i],complex_words[i],args.beam, args.bertscore)

       
        print(bert_substitutes)
        print(bert_rank_substitutes)
 
        #print(rank_substitutes)

            
       
        CS2.append(bert_substitutes[:10])
        CS3.append(bert_rank_substitutes[:10])


        #break
        #if i==5:
            #break
    
    output_sr_file.write(args.paraphraser_model)
    output_sr_file.write('\n')
   
    potential,precision,recall,F_score=evaulation_SS_scores(CS2, complex_labels)
    print("The score of evaluation for candidate selection")
    output_sr_file.write(str(potential))
    output_sr_file.write('\t')
    output_sr_file.write(str(precision))
    output_sr_file.write('\t')
    output_sr_file.write(str(recall))
    output_sr_file.write('\t')
    output_sr_file.write(str(F_score))
    output_sr_file.write('\n')
    print(potential,precision,recall,F_score)
    potential,precision,recall,F_score=evaulation_SS_scores(CS3, complex_labels)
    print("The score of evaluation for candidate selection")
    output_sr_file.write(str(potential))
    output_sr_file.write('\t')
    output_sr_file.write(str(precision))
    output_sr_file.write('\t')
    output_sr_file.write(str(recall))
    output_sr_file.write('\t')
    output_sr_file.write(str(F_score))
    output_sr_file.write('\n')
    print(potential,precision,recall,F_score)
   
    
    output_sr_file.close()

if __name__ == "__main__":
    main()

