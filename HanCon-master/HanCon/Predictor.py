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

from HanCon import EmbeddingModel


class Predictor:
    def __init__(self, emb=None):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_m = os.path.join(my_path, "models/")
        path_model = os.path.join(path_m, "model_ft_concreteness.pkl")
        path_porter = os.path.join(path_m, "most_common_porter.pkl")
        path_ending = os.path.join(path_m, "most_common_ending.pkl")
        if (not os.path.isfile(path_model)):
            print("HanCon-Model not found!\nDownloading HanCon-Model ...")
            self.downloadHanConModel(path_m)
            print("HanCon-Model Download Done!")
        else:
            print("HanCon-Model found!")
        self.trained_model = pickle.load(open(path_model, 'rb'))

        '''Constructor'''
        print("Init")
        self.model = emb
        # self.trained_model=pickle.load(open("models/model_ft_final4.pkl", 'rb'))

        self.stemmer = PorterStemmer()
        with open(path_porter, 'rb') as f:
            self.most_common_ending = pickle.load(f)
        with open(path_ending, 'rb') as f:
            self.most_common_porter = pickle.load(f)

    def downloadHanConModel(self, path):
        url = "http://textmining.wp.hs-hannover.de/datasets/model_ft_concreteness.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)

    def getFeature(self, w, emb=True, POS=True, Post=True, verbose=0):
        vect = []
        vect = np.asarray(vect)
        if verbose > 0:
            print(feature_vector)
        if emb:
            vect = np.append(vect, self.model[w])
        if POS:
            vect = np.append(vect, self.getPos(w))
        if Post:
            vect = np.append(vect, self.getPostfixVec(w, self.most_common_ending, dim=200))
        return vect

    def getPrediction(self, vec, trained_model):
        return trained_model.predict([vec])

    def predict(self, word):
        try:
            features = self.getFeature(word)
        except KeyError:
            print("[ERROR]:\"", word, "\" is not in the fastText Vocabulary")
            return 0
        return self.getPrediction(features, self.trained_model)

    def getAllPostfix(self, s):
        if len(s) > 3:
            return [s[-1:], s[-2:], s[-3:], s[-4:]]
        elif len(s) == 3:
            return [s[-1:], s[-2:], s[-3:]]
        elif len(s) == 2:
            return [s[-1:], s[-2:]]
        elif len(s) == 1:
            return [s]
        else:
            return [""]

    def getPostfixVec(self, s, mc, dim=100):
        vec = [0] * dim
        Post = self.getAllPostfix(s)
        # Post=Func(s)
        for i in range(dim):
            if mc[i] in Post:
                vec[i] = 1
        return vec

    def getPorterVec(self, s, mc, dim=150):
        vec = [0] * dim
        Post = self.getPorterRest(s)
        print(mc)
        for i in range(dim):
            if mc[i] == Post:
                vec[i] = 1
        return vec

    def getPorterRest(self, s):
        stem = self.stemmer.stem(s)
        ret = ""
        for k in range(len(s) - len(stem)):
            ret += (s[len(stem) + k])
        return ret

    def getPos(self, w):
        poss = []  # this was two lines above, globel
        posSum = 0
        posfreqs = {'n': 0, 'v': 0, 'a': 0, 'p': 0, 'r': 0, 's': 0}
        found = False
        for syns in wn.synsets(w):
            p = syns.pos()
            if p not in poss:
                poss.append(p)
            count = posfreqs.get(p, 0)
            for l in syns.lemmas():
                count += l.count()
                if count > 0:
                    found = True
            posfreqs[p] = count
        if found:
            for fr in posfreqs:
                posSum += posfreqs[fr]
            pos = [posfreqs['n'] / posSum, posfreqs['v'] / posSum, posfreqs['a'] / posSum, posfreqs['p'] / posSum,
                   posfreqs['r'] / posSum, posfreqs['s'] / posSum]
        else:
            tags = nltk.pos_tag([w])
            tag = tags[0][1]
            if tag[0] == 'R':
                pos = 'r'
                pos = [0, 0, 0, 0, 1, 0]
            elif tag[0] == 'J':
                pos = 'a'
                pos = [0, 0, 1, 0, 0, 0]
            elif tag[0] == 'V':
                pos = 'v'
                pos = [0, 1, 0, 0, 0, 0]
            elif tag == 'IN':
                pos = 'p'
                pos = [0, 0, 0, 1, 0, 0]
            else:
                pos = 'n'
                pos = [1, 0, 0, 0, 0, 0]
        return pos


EmbModel = EmbeddingModel()

predictor = Predictor(EmbModel.emb)

concreteness_value = predictor.predict("word")
