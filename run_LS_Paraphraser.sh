
##NNSeval
#python3 Substitute_Para.py \
#  --eval_dir "data/swords-v1.1_test.json" \
#  --paraphraser_path "checkpoints/para/transformer/" \
#  --paraphraser_model "checkpoint_best.pt" \
#  --bpe "subword_nmt" \
#  --bpe_codes "data/para/codes.40000.bpe.en" \
#  --paraphraser_dict 'data-bin/para/' \
#  --output_SR_file "results/swords-v1.1_test_mygenerator.lsr.json"  


python3 LSPara.py \
  --eval_dir "data/NNSeval.txt" \
  --paraphraser_path "checkpoints/para/transformer/" \
  --paraphraser_model "checkpoint_best.pt" \
  --bpe "subword_nmt" \
  --bpe_codes "data/para/codes.40000.bpe.en" \
  --paraphraser_dict 'data-bin/para/' \
  --beam 20 \
  --bertscore 0.7  \
  --output_SR_file "results/suffix_bert/LSPara-NNSeval.txt"  
