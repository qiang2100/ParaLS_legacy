
##NNSeval
#python3 Substitute_Para.py \
#  --eval_dir "data/swords-v1.1_test.json" \
#  --paraphraser_path "checkpoints/para/transformer/" \
#  --paraphraser_model "checkpoint_best.pt" \
#  --bpe "subword_nmt" \
#  --bpe_codes "data/para/codes.40000.bpe.en" \
#  --paraphraser_dict 'data-bin/para/' \
#  --output_SR_file "results/swords-v1.1_test_mygenerator.lsr.json"  

lex=lex.mturk.txt
nnsevl=NNSeval.txt
bench=BenchLS.txt



beam=30
for bert in 0.6 0.65 0.7 0.75 0.8 0.85 0.9

do

  python3 LSPara.py \
    --eval_dir data/$nnsevl \
    --paraphraser_path "checkpoints/para/transformer/" \
    --paraphraser_model "checkpoint_best.pt" \
    --bpe "subword_nmt" \
    --bpe_codes "data/para/codes.40000.bpe.en" \
    --paraphraser_dict 'data-bin/para/' \
    --beam $beam \
    --bertscore $bert  \
    --output_SR_file "results/LSPara-NNSeval.txt" 
done

bert=0.7

for beam in 10 15 20 25 30 35 40 45 50

do

  python3 LSPara.py \
    --eval_dir data/$nnsevl \
    --paraphraser_path "checkpoints/para/transformer/" \
    --paraphraser_model "checkpoint_best.pt" \
    --bpe "subword_nmt" \
    --bpe_codes "data/para/codes.40000.bpe.en" \
    --paraphraser_dict 'data-bin/para/' \
    --beam $beam \
    --bertscore $bert  \
    --output_SR_file "results/LSPara-NNSeval.txt" 
done