import pickle
import gensim
import io
import nltk
import requests
import numpy as np
import zipfile
import os
from math import *
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

class EmbeddingModel:
    def __init__(self,ft_model="crawl-300d-2M.vec"):
        self.fname = get_tmpfile("fastText_vectors.kv")

        if(os.path.isfile('./wordEmbeddings/crawl-300d-2M.vec')):
            print("Embedding Found!\nLoading Embeddings with mmap." )
            self.model = KeyedVectors.load(self.fname, mmap='r')
            print("Done!")
            #self.emb=KeyedVectors.load_word2vec_format("./wordEmbeddings/"+ft_model
        else:
            print("Missing FastText Vectors at \"wordEmbeddings/\" \nDownloading now!... This will take some time (and 4,5GB space) ! ")
            self.downloadFT()
            
            print("Loading Embeddings, this will ALSO take some time! (until mmap)" )
            #fname = get_tmpfile("fastText_vectors.kv")
            #model = KeyedVectors.load(fname, mmap='r')
            self.model=KeyedVectors.load_word2vec_format("./wordEmbeddings/"+ft_model)
            self.model.save(self.fname)
            #self.model = KeyedVectors.load(self.fname, mmap='r')
            print("mmap created!")


    def downloadFT(self):
        url="https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./wordEmbeddings/")
        print("fastText Download Done!")
