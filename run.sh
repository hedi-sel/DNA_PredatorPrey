#!/bin/bash
cd $(dirname $0)

rm -f ./dataName;
make && 
run/predatorPrey && 
dataName=$(cat dataName) && 
rm -f ./dataName &&
python3 Ploter.py $dataName