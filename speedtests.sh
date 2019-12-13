#!/bin/bash
cd $(dirname $0)

make && 
for i in {1..10}
do
    run/predatorPrey gpu
done