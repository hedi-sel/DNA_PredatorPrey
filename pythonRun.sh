#!/bin/bash
cd $(dirname $0)

cd build &&
make && 
cp dna.so ../src/ &&
cd ../ &&
python3 src/run.py

