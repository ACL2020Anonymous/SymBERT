# SymBERT

This README contains the instructions for "Symmetric Regularization based BERT for Pair-wise Semantic Reasoning".

## Pretrained Models:

- The pretrained Chinese BERT models:
    - PyTorch:
        - BERT-PN: [Download](https://drive.google.com/open?id=1q2WerrRTSFkKIcA51Ftb1UfgjuD-MXLi)
        - BERT-PNsmth: [Download](https://drive.google.com/open?id=1vIWg-LJUenjccPkGymZT1gq3TvBr6Qqx)
    - Tensorflow:
        - BERT-PN: [Download](https://drive.google.com/open?id=1q2WerrRTSFkKIcA51Ftb1UfgjuD-MXLi)
        - BERT-PNsmth: [Download](https://drive.google.com/open?id=1piBkehIh3tpZybghETZkzcc9a7Sl98yW)


## Usage

This code is based on the BERT repository.
To train the model, e.g., BERT-PNsmth, you should first process the dataset as follows to obtain the tfrecords.

````
DATA_DIR="data/zhwiki_files_sharded_simple"
MODEL_DIR="models/chinese_L-12_H-768_A-12/"
EXPNAME="wiki-zh-bert-pretrain-3clssmooth"
OUTPUT_DIR="data/"${EXPNAME}

mkdir -p ${OUTPUT_DIR}

ls ${DATA_DIR} | awk -F"." '{print $3}' | awk -F"part" '{print $2}' | xargs -n 1 -P 25 -I % -t python create_pretraining_data.py \
        --input_file=${DATA_DIR}/wikipedia.segmented.part%.txt \
        --output_file="${OUTPUT_DIR}"/%.tfrecord \
        --vocab_file=${MODEL_DIR}/vocab.txt \
        --max_seq_length=128 \
        --max_predictions_per_seq=20 \
        --dupe_factor=5 > ${EXPNAME}.log 2>&1 &

EXPNAME="wiki-zh-bert-pretrain-3clssmooth-512"
OUTPUT_DIR="data/"${EXPNAME}

mkdir -p ${OUTPUT_DIR}

ls ${DATA_DIR} | awk -F"." '{print $3}' | awk -F"part" '{print $2}' | xargs -n 1 -P 25 -I % -t python create_pretraining_data.py \
        --input_file=${DATA_DIR}/wikipedia.segmented.part%.txt \
        --output_file="${OUTPUT_DIR}"/%.tfrecord \
        --vocab_file=${MODEL_DIR}/vocab.txt \
        --max_seq_length=512 \
        --max_predictions_per_seq=80 \
        --dupe_factor=5 > ${EXPNAME}.log 2>&1 &
````
`zhwiki_files_sharded_simple` can be [downloaded]().

Then run the train script:

````
sh train.sh
````
,which requires [hovorod]() as the distribution optimization tool.

## Experimental Results

Empirical results for XNLI (zh part), LCQMC and NLPCC-DBQA:

|                                        | metrics  | BERT        |    BERT-wwm |    ERNIE |          ERNIE2.0 |             NEZHA |              BERT-PN |                   BERT-PNsmth |
|----------------------------------------|----------|-------------|------------:|---------:|------------------:|------------------:|---------------------:|------------------------------:|
|                                        |          | 393M        |        393M |        - |            14988M |            10536M |               10879M |                        10879M |
| dev  |          |             |             |          |                   |                   |                      |                               |
| XNLI                                   | Accuracy | 77.8 (77.4) | 79.0 (78.4) | - (79.9) |          - (81.2) | - (**81.3**) |          80.5 (79.9) |          **81.4** (81.0) |
| LCQMC                                  | Accuracy | 89.4 (88.4) | 89.4 (89.2) | - (89.7) | - (**90.9**) |          - (89.9) |          90.3 (89.4) |          **90.6** (90.1) |
| NLPCC-DBQA                             | F1       | - (80.7)    |       - (-) | - (82.3) |          - (84.7) |             - (-) |          85.0 (84.6) | **85.9** (**85.4**) |
| test |          |             |             |          |                   |                   |                      |                               |
| XNLI                                   | Accuracy | 77.8 (77.5) | 78.2 (78.0) | - (78.4) |          - (79.7) |          - (79.1) |          79.8 (79.4) | **80.3** (**79.9**) |
| LCQMC                                  | Accuracy | 86.9 (86.4) | 87.0 (86.8) | - (87.4) |          - (87.9) |          - (87.1) | **88.7** (87.5) | **88.7** (**88.0**) |
| NLPCC-DBQA                             | F1       | - (80.8)    |       - (-) | - (82.7) |          - (85.3) |             - (-) |          85.2 (84.9) | **86.2** (**85.9**) |

Results for CMRC-2018 and DRCD.

|                        |        CMRC-2018 (Dev)        |                   |           DRCD (Dev)          |                               |          DRCD (Test)          |                               |
|------------------------|:-----------------------------:|------------------:|:-----------------------------:|------------------------------:|:-----------------------------:|------------------------------:|
| metrics                | F1                            |                EM |                            F1 |                            EM |                            F1 |                            EM |
| BERTBase (ours)        | 84.7 (84.3)                   |       64.1 (63.8) |                   90.2 (90.0) |                   83.5 (83.4) |                   89.0 (88.9) |                   82.0 (81.8) |
| BERTBase (wwm)    | 84.5 (84.0)                   |       65.5 (64.4) |                   89.9 (89.6) |                   83.1 (82.7) |                   89.2 (88.8) |                   82.2 (81.6) |
| BERTBase (ERNIE2) | - (85.9)                      |          - (66.3) |                      - (91.6) |                      - (85.7) |                      - (90.9) |                      - (84.9) |
| BERTBase-wwm           | 85.6 (84.7)                   |       66.3 (65.0) |                   90.5 (90.2) |                   83.7 (83.5) |                   89.8 (89.4) |                   82.7 (82.1) |
| NEZHA-wwm              | - (86.3)                      | - (\textbf{67.8}) |                         - (-) |                         - (-) |                         - (-) |                         - (-) |
| BERTBase-PN            | **87.5** (**86.8**) |       66.6 (65.8) |                   92.3 (92.0) |                   86.4 (86.0) |                   92.3 (92.2) |                   86.1 (86.0) |
| BERTBase-PNsmth        | 86.4 (86.2)                   |       66.5 (66.3) | **93.0** (**92.7**) | **86.8** (**86.8**) | **92.6** (**92.5**) | **86.7** (**86.6**) |
