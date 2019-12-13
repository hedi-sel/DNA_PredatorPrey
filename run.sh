#!/bin/bash
cd $(dirname $0)

rm -f ./dataName;
cd build &&
make && 
cd .. &&
run/predatorPrey && 
dataName=$(cat dataName) && 
rm -f ./dataName &&
python3 Ploter.py $dataName