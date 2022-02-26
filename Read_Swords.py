from collections import defaultdict
import gzip
import json
import warnings
from nltk import tokenize

# Load benchmark
with open('data/swords-v1.1_test.json', 'r') as f:
  swords = json.load(f)

with open('data/swords-v1.1_test_LSPara_20_0.7.json', 'r') as f:
  ParaLS = json.load(f)

with open('data/swords-v1.1_test_LSPara_20_direct.json', 'r') as f:
  ParaLS2 = json.load(f)

with open('data/swords_test2_LSBert.result.json', 'r') as f:
  BertLS = json.load(f)

with open('data/swords_test_gpt3.result.json', 'r') as f:
  GPT3 = json.load(f)

with open('data/swords_test_wordtune.result.json', 'r') as f:
  WordTune = json.load(f)


# Gather substitutes by target
tid_to_sids = defaultdict(list)
for sid, substitute in swords['substitutes'].items():
  tid_to_sids[substitute['target_id']].append(sid)

# Iterate through targets

i = 0
for tid, target in swords['targets'].items():

  #if len(target['target'])<10:
  #  continue

  print('Sentence {} rankings: '.format(i))



  context = swords['contexts'][target['context_id']]
  substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
  substitutes_ParaLS = ParaLS['substitutes'][tid]
  substitutes_ParaLS2 = ParaLS2['substitutes'][tid]
  substitutes_BertLS= BertLS['substitutes'][tid]
  substitutes_GPT3= GPT3['substitutes'][tid]

  if tid not in WordTune['substitutes']:
    continue
  substitutes_WordTune = WordTune['substitutes'][tid]
  #print(substitutes_ParaLS)
  labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
  scores = [l.count('TRUE') / len(l) for l in labels]
  print('-' * 80)
  
  print(context['context'])
  print(context['context'][target['offset']:target['offset']+len(target['target'])])
  print('{} {} ({})'.format(target['offset'], target['target'], target['pos']))
  print('-' * 20)
  #print('{} {} ({})'.format(target['offset'], target['target'], target['pos']))
  #print(substitutes)
  print(', '.join(['{} ({}%)'.format(substitute['substitute'], round(score * 100)) for substitute, score in sorted(zip(substitutes, scores), key=lambda x: -x[1])]))

  print("ParaLS:")
  print(', '.join(['{}'.format(substitute[0]) for substitute in substitutes_ParaLS]))
  print("ParaLS:")
  print(', '.join(['{}'.format(substitute[0]) for substitute in substitutes_ParaLS2]))
  print("BertLS:")
  print(', '.join(['{}'.format(substitute[0]) for substitute in substitutes_BertLS]))
  print("GPT3:")
  print(', '.join(['{}'.format(substitute[0]) for substitute in substitutes_GPT3]))
  print("WordTune:")
  print(', '.join(['{}'.format(substitute[0]) for substitute in substitutes_WordTune]))
  
  

  i += 1

  #if i==10:
  #  break



