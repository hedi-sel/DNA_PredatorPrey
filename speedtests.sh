#!/bin/bash

make && 
for i in {1..10}
do
    run/predatorPrey cpu
done
echo " "
for i in {1..10}
do
    run/predatorPrey gpu
done