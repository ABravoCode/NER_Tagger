# NER_Tagger
Assignment 2 of COMP7607, HKU, 2023

## Introduction

In Natural Language Processing (NLP), Named Entity Recognition (NER) serves as a critical foundation for many applications, from information retrieval to question-answering systems. NER is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

This repository implements two model for NER task, namely LSTM model and Transformer model.

## Contacts

Email: [jeremyan@connect.hku.hk](jeremyan@connect.hku.hk)

## To run this code

First, build the enviroment by:

```shell
cd ./codes
conda create -n ner python==3.9
conda activate ner
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r reqirements.txt
```

After you complete it, run:

```shell
bash run_lstm.sh
```

or

```shell
bash run_transformer.sh
```

for reproducing.

## Reference from author
senadkurtisi
baaraban
pauljhp
Caffretro
