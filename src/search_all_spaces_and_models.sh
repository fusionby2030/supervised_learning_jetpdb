#!/bin/sh

search_configs=$(ls ./src/configs/search/*.yaml | xargs -n 1 basename)
for config in $search_configs
do 
    for value in 1 2 3 4 5 6 7 8 
    do
        python3 search.py -c $config -nit 500 -is $value 
    done 
done