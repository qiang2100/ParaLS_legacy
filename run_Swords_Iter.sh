

path=/home/nlp/Desktop/swords/


bert=0.7

for beam in 45 50

do

  python3 LSPara_SwordS.py \
    --eval_dir "data/swords-v1.1_test.json" \
    --paraphraser_path "checkpoints/para/transformer/" \
    --paraphraser_model "checkpoint_best.pt" \
    --bpe "subword_nmt" \
    --bpe_codes "data/para/codes.40000.bpe.en" \
    --paraphraser_dict 'data-bin/para/' \
    --beam $beam \
    --bertscore $bert  \
    --output_SR_file $path/swords-v1.1_test_LSPara_${beam}_${bert}_multi_suffix.json
done


beam=30

for bert in 0.6 0.65 0.7 0.75 0.8 0.85 0.9

do

  python3 LSPara_SwordS.py \
    --eval_dir "data/swords-v1.1_test.json" \
    --paraphraser_path "checkpoints/para/transformer/" \
    --paraphraser_model "checkpoint_best.pt" \
    --bpe "subword_nmt" \
    --bpe_codes "data/para/codes.40000.bpe.en" \
    --paraphraser_dict 'data-bin/para/' \
    --beam $beam \
    --bertscore $bert  \
    --output_SR_file $path/swords-v1.1_test_LSPara_${beam}_${bert}_multi_suffix.json

done


