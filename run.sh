#!/bin/bash

rm -f ./dataName;
make && 
run/predatorPrey && 
dataName=$(cat dataName) && 
rm -f ./dataName &&
python3 Ploter.py $dataName