#!/bin/bash
cd $(dirname $0)

cd build &&
make && 
cp dna.so ../src/ &&
cd ../ &&
python2.7 src/run.py

