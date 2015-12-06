#! /bin/bash

mkdir -p dataset
cd dataset
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
cd ..
./convert.py
