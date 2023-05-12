# Modeling-SARS-CoV-2-Viral-Escape-Mutations

## Description
This directory comprises the folders containing the source code, data and figures for the research article: 
> _Modeling-SARS-CoV-2--Viral-Escape-Mutations-with-Natural-Language-Processing-Methods_

## Source
This folder contains the python source code for analysing the mutation dynamics of the SARS-CoV-2 genome:

> seq.py

* This python script contains the supporting functions for pre-processing of the raw genome sequences

> spatial.py

* This python script contains the main driver code for our spatial model and analysis

> time_series.py

* This python script contains the main driver code for our time-series model and analysis

## Data
This folder contains the genome sequence data exported from GISAID EpiFlu / NCBI Genbank database:
* Each data file is provided in a variety of readable text formats (_.fasta, .aln, .txt_)
* Text header contains relevant metadata on each sample (_sequence ID, collection date, country of origin_)
* Text body contains a string of characters representing the corresponding RNA nucleotides (_A, G, T, C_)

> **Sample data file (truncated):**   
>           
> ![](figures/image6.png)

## Figures
This folder contains all the output figures generated from our analyses.

## How to Run the Code:
* Download the _spatial.py_, _time-series.py_, _seq.py_ python scripts from _source_ folder and place them in a directory. 
* Download the sequence text files from _data_ folder and place them in the same directory as the python scripts.
* Run the scripts _spatial.py_ and _time-series.py_, then check your directory for the output figures.
