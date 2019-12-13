#!/bin/bash
cd $(dirname $0)

cd build &&
make && 
cp dna.so &&
cd ../src &&
python run.py

