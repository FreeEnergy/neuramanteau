# Neuramanteau
Accompanying code for the paper -  Neuramanteau: A Neural Network Ensemble Model for Lexical Blends.

Try out the demo [here](https://neuroblender.herokuapp.com)

## Requirements
* Python 3.5+
* Tensorflow 1.1+
* Numpy 1.11+

## Usage
1. Train the model  
    * `python blender.py train --data wiktionary-blends-shuffled.csv --num-epochs 16`
2. Evaluate the model  
    a. Validation set:  `python blender.py eval --data wiktionary-blends-shuffled.csv --partition-name validation`  
    b. Test set:  `python blender.py eval --data wiktionary-blends-shuffled.csv --partition-name test`  
    c. External dataset:  `python blender.py eval --data <dataset csv file> --no-partitions`  
3. Sample outputs:  
    * `python blender.py sample --data wiktionary-blends-shuffled.csv --source-words work alcoholic`
    
By default the model uses 6 experts. To change the number of experts use the flag `--num-experts 15` (for 15 experts). You must pass the same values during sampling and evaluation.

To use dominances add the flag `--add-dominances` during training. During sampling set dominances via `--dominances 1 0` (1st word dominant).

## Dataset
The blend dataset we provide here was scraped from [Wiktionary](https://en.wiktionary.org/wiki/Category:English_blends) on March 2017. It consists of 3250 blends (unfiltered). Each line of the dataset contains the blends in the following format:

*target blend, first source word, second source word*

  
Three CSV files are provided:
* *wiktionary-blends-shuffled.csv* : The entire blend dataset
* *wiktionary-blends-shuffled-cs.csv*: Contains subset of the blend dataset which have overlapping characters (common string)
* *wiktionary-blends-shuffled-nocs.csv*: Contains subset of the blend dataset which have no overlapping characters

*DISCLAIMER: The above dataset is provided as is and we are not liable for its contents* 
