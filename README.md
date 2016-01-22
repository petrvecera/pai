# PA I
School project for subject Parallel algorithms. Text similarity. Written in Python with usage of MPI.


### Installation
1. python3
2. pip install mpi4py
3. mpi (mpiexec has to be available)


### Usage
1. Enter the paths for text files into the file
2. For Single thread Run `python mainsp.py filelist.txt`
3. For MPI run: `mpiexec -n 5 python mainmpi.py filelist.txt` 


### Overview of algorithm TF-IDF and Cosine Similarity
1. TF for each document
2. IDF for all words across all documents
3. TF*IDF vectors for each document
4. Cosine similarity of the vectors (documents) 



### Example output
```
Input file not specified trying with default:
Usage: file.py filelist.txt
Data load time          --- 0.040 seconds ---
Loaded files:
File 1: lotr1.txt
File 2: lotr2.txt
File 3: twk.txt
File 4: Dune.txt


Text 1-1: 1.000
Text 1-2: 0.991
Text 1-3: 0.904
Text 1-4: 0.906

Text 2-1: 0.991
Text 2-2: 1.000
Text 2-3: 0.906
Text 2-4: 0.904

Text 3-1: 0.904
Text 3-2: 0.906
Text 3-3: 1.000
Text 3-4: 0.903

Text 4-1: 0.906
Text 4-2: 0.904
Text 4-3: 0.903
Text 4-4: 1.000

TF compute time         --- 0.996 seconds ---
IDF compute time        --- 0.066 seconds ---
TF*IDF compute time     --- 0.029 seconds ---
Cos Sim compute time    --- 0.171 seconds ---
Complete compute time   --- 1.313 seconds ---
```



### Sources
https://en.wikipedia.org/wiki/Tf%E2%80%93idf
https://en.wikipedia.org/wiki/Cosine_similarity
https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/