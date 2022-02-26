

import pdb
from fairseq.models.transformer import TransformerModel
import torch

en2en = TransformerModel.from_pretrained(
  'checkpoints/para/transformer/',
  checkpoint_file='checkpoint_best.pt',
  bpe='subword_nmt',
  bpe_codes='checkpoints/para/transformer/codes.40000.bpe.en'
)


s9 = 'The books were adapted into a television series .'

s19 = 'The books were'

prefix_tokens = en2en.encode(s19)
prefix_tokens = prefix_tokens[:-1]
print(prefix_tokens)

complex_tokens = en2en.encode('adapted')
print(complex_tokens)

#print(type(prefix_tokens))


en = en2en.encode(s9)


#pdb.set_trace()

prefix_tokens = prefix_tokens.view(1,-1)

#print(len(prefix_tokens[0]))



attn_len = len(prefix_tokens[0])+len(complex_tokens)-1

outputs,sss = en2en.generate2(en, beam=8, prefix_tokens=prefix_tokens, attn_len=attn_len)
#outputs = en2en.generate(en, beam=12)

#print(outputs)

#for x in outputs:
#  print(x['tokens'])

two = [en2en.decode(x['tokens']) for x in outputs]

#curr_tokens = [en2en.decode(x['tokens']) for x in curr_indexs]

for sen in two:
  print(sen)


#three = [s2,s3,s4,s5,]
#print(three)
#ss = en2en.score(three)
#two = [en2en.decode(x['tokens']) for x in ss]
#print(two)




