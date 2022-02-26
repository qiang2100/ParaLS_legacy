python3 LSPara_SwordS.py \
  --eval_dir "data/swords-v1.1_test.json" \
  --paraphraser_path "checkpoints/para/transformer/" \
  --paraphraser_model "checkpoint_best.pt" \
  --bpe "subword_nmt" \
  --bpe_codes "checkpoints/para/transformer/codes.40000.bpe.en" \
  --paraphraser_dict 'checkpoints/para/transformer/' \
  --beam 20 \
  --bertscore 0.7  \
  --output_SR_file "/home/nlp/Desktop/swords/swords-v1.1_test.json"

