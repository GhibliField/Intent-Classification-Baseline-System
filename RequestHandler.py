# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by W.Y Shen on 2018/4/18
"""
from __future__ import print_function
import sys;sys.path.append("..")
from config import configs
from utils.utils import ACAutomatons, get_predicate, get_nouns, doRetrieval, get_rule, getSRL, aggregate_domains, prettySureExpression
import gc
from pyltp import SementicRoleLabeller
from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
from trainers.classifiers import NBSVM
import jieba.posseg as pseg
import jieba
from scipy.sparse import  hstack
from sklearn.externals import joblib
from scipy.sparse import csr_matrix


class RequestHandler():

    def __init__(self):
        self.intents=['translation',
             'app',
             'calc',
             'match',
             'radio',
             'health',
             'novel',
             'video',
             'cinemas',
             'music',
             'stock',
             'train',
             'news',
             'message',
             'map',
             'weather',
             'cookbook',
             'tvchannel',
             'flight',
             'schedule',
             'riddle',
             'email',
             'contacts',
             'bus',
             'website',
             'datetime',
             'poetry',
             'lottery',
             'chat',
             'epg',
             'telephone']

        self.segmentor = Segmentor()  # 初始化实例 CWS
        self.segmentor.load(configs.cws_path)  # 加载模型
        self.postagger = Postagger() # 初始化实例 POS Tagger
        self.postagger.load(configs.pos_path)  # 加载模型
        self.labeller = SementicRoleLabeller() # 初始化实例 SRLer
        self.labeller.load(configs.srl_path)  # 加载模型
        self.parser = Parser() # 初始化实例   Parser
        self.parser.load(configs.parser_path)  # 加载模型
        
        self.ac=ACAutomatons()

        self.clf_31=NBSVM()

        self.char_vectorizer_31=joblib.load(configs.models_path+'/nbsvm-vocab-ch.pkl')
        self.word_vectorizer_31=joblib.load(configs.models_path+'/nbsvm-vocab-wd.pkl')
        self.clf_31=joblib.load(configs.models_path+'/nbsvm_31.pkl')
        self.ch2_=joblib.load(configs.models_path+'/nbsvm-feature_selector.pkl')
        self.word_vectorizer_tv=joblib.load(configs.models_path+'/vocab-wd_epg-tvchannel.pkl')
        self.char_vectorizer_tv=joblib.load(configs.models_path+'/vocab-ch_epg-tvchannel.pkl')
        self.clf_tv=joblib.load(configs.models_path+'/svm_epg-tvchannel.pkl')
        self.word_vectorizer_movie=joblib.load(configs.models_path+'/vocab-wd_video-cinemas.pkl')

        self.char_vectorizer_movie=joblib.load(configs.models_path+'/vocab-ch_video-cinemas.pkl')
        self.clf_movie=joblib.load(configs.models_path+'/svm_video-cinemas.pkl')
        self.char_vectorizer_internet=joblib.load(configs.models_path+'/vocab-ch_website-app.pkl')
        self.word_vectorizer_internet=joblib.load(configs.models_path+'/vocab-wd_website-app.pkl')
        self.clf_internet=joblib.load(configs.models_path+'/svm_website-app.pkl')
        self.char_vectorizer_star=joblib.load(configs.models_path+'/vocab-ch_video-music.pkl')
        self.clf_star=joblib.load(configs.models_path+'/svm_video-music.pkl')

        self.word_vectorizer_star=joblib.load(configs.models_path+'/vocab-wd_video-music.pkl')
        self.char_vectorizer_video=joblib.load(configs.models_path+'/vocab-ch_video-epg.pkl')
        self.word_vectorizer_video=joblib.load(configs.models_path+'/vocab-wd_video-epg.pkl')
        self.clf_video=joblib.load(configs.models_path+'/svm_video-epg.pkl')


    def getResult(self, sentence):
        """1. Complete the classification in this function.

        Args:
            sentence: A string of sentence.

        Returns:
            classification: A string of the result of classification.
        """
        processed=self.preprocess(sentence)

        return self.pipeline(processed)

    def getBatchResults(self, sentencesList):
        """2. You can also complete the classification in this function,
                if you want to classify the sentences in batch.

        Args:
            sentencesList: A List of Dictionaries of ids and sentences,
                like:
                [{'id':331, 'content':'帮我打电话给张三' }, 
                 {'id':332, 'content':'帮我订一张机票!' },
                 ... ]

        Returns:
            resultsList: A List of Dictionaries of ids and results.
                The order of the list must be the same as the input list,
                like:
                [{'id':331, 'result':'telephone' }, 
                 {'id':332, 'result':'flight' },
                 ... ]
        """
        resultsList = []
        for sentence in sentencesList:
            resultDict = {}
            resultDict['id'] = sentence['id']
            resultDict['result'] = self.getResult(sentence['content'])
            resultsList.append(resultDict)

        return resultsList


    def pattern_match(self,sample):
        srl_res=self.sRLMatch(sample)
        if srl_res!=None:

            return srl_res
        else:
            rul_res=self.ruleMatch(sample)
            if rul_res!=None:

                return rul_res
            else:
                return None


    def ruleMatch(self,sample):
        domains = get_rule(sample['query'],self.ac)

        if len(domains) <1:
            return None
        else:
            sorted_domains=aggregate_domains(domains)

            for each in sorted_domains:
                if each[0]=='datetime':
                    nouns=get_nouns(sample['query'],'festival',self.ac)

                    if len(nouns)>0:
                        return 'datetime'
                    else:
                        continue

                elif each[0]=='email':
                    if len(set(sample['word'])& set(['写','回复','转发','打开','查收','查看','答复'])) >0:
                        return 'email'
                    else:
                        continue

            else:return None


    def sRLMatch(self,sample):
        srl_res=getSRL(sample['query'],self.segmentor,self.postagger,self.parser,self.labeller)
        if len(srl_res)==0:#no any predicate in query or single entity
            return None
        else:
            for res in srl_res:
                predicate_domains = get_predicate(res[0],self.ac)
                if len(predicate_domains) <1:
                    continue#no such a predicate in database
                else:
                    sorted_domains=aggregate_domains(predicate_domains)
                    for each in sorted_domains:
                        if each[0]=='app':
                            nouns = get_nouns(res[1], 'app',self.ac)
                            if len(nouns) > 0 :
      
                                return 'app'
                            else:
                                continue

                        elif each[0]=='cinemas':
                            nouns=get_nouns(res[1],'film',self.ac)
                            if len(nouns)>0:
                                return 'Movie_stuff'
                            else:
                                continue
                        elif each[0]=='contacts':
                        # 'nr' by POS-tagger indicates a person's name
                            if 'nr' in sample['tag'] :
                                return 'contacts'
                            else:
                                continue


                        elif each[0]=='cookbook':
                            nouns = get_nouns(res[1], 'food',self.ac)
                            if len(nouns) > 0:  # 如果命中任何专有名词，则划分到意图app

                                return 'cookbook'
                            else:
                                continue

                        elif each[0]=='tvchannel':
                            nouns = get_nouns(res[1], 'tvchannel',self.ac)
                            if len(nouns) > 0 :
                                return 'TV_stuff'
                            else:
                                continue

                        elif each[0]=='video':
                            nouns = get_nouns(res[1], 'video',self.ac)
                            if len(nouns) > 0 :
                                return 'Video_stuff'
                            else:
                                continue
       
                        elif each[0]=='health':
                            nouns = get_nouns(res[1], 'disease',self.ac)
                            nouns.extend(get_nouns(res[1], 'drug',self.ac))
                            if len(nouns) > 0:
                                return 'health'
                            else:
                                continue

                        elif each[0]=='music':
                            nouns_song = get_nouns(res[1], 'song',self.ac)
                            nouns_singer=get_nouns(res[1], 'singer',self.ac)
                            if len(nouns_song) > 0:
                                
                                return 'music'
                            elif len(nouns_singer)>0:
                                return 'Star_stuff'
                            else:
                                continue
                        
                        elif each[0]=='novel':
                            nouns = get_nouns(res[1],'novel',self.ac)
                            if '小说' in res[1] or len(nouns)>0:

                                return 'novel'
                            else:
                                continue

                        elif each[0]=='poetry':
                            nouns = get_nouns(res[1], 'poet',self.ac)
                            if len(nouns) > 0:
                               
                                return 'poetry'
                            else:
                                continue

                        elif each[0]=='radio':
                            if len(get_nouns(res[1],'radio',self.ac))>0:

                                return 'radio'
                            else:
                                continue

                        elif each[0]=='stock':
                            nouns=get_nouns(res[1],'stock',self.ac)
                            if len(nouns)>0:

                                return 'stock'
                            else:
                                continue

                        elif each[0]=='website':
                            nouns=get_nouns(res[1],'website',self.ac)
                            if len(nouns)>0:

                                return 'Internet_stuff'
                            else:
                                continue


    def retrieval(self,sample):
        """
        To find proper nouns to handle single entity in a query
        :param sample: a dict indicates a query and its POS tag
        :return:a string indicates one certain intent
        """
        pn_res=doRetrieval(sample['query'],self.ac)#look up single instance
        sorted_domains=aggregate_domains(pn_res)
        if len(sorted_domains)==1:#one instance
            domain=sorted_domains[0][0]
            if len(max(sorted_domains[0][1], key = len))>len(sample['query'])/2:
                if domain == 'airline':return 'flight'
                if domain in ['railwaystation', 'airport']:return 'map'
                if domain == 'app':return 'app'
                if domain == 'contacts':return 'contacts'
                if domain in ['drug', 'disease']:return 'health'
                if domain == 'festival':return 'datetime'
                if domain in ['moviestar', 'film','video']:return 'video'
                if domain == 'food':return 'cookbook'
                if domain == 'novel':return 'novel'
                if domain == 'place':return 'map'
                if domain == 'poet':return 'poetry'
                if domain == 'radio':return 'radio'
                if domain in ['singer','song']:return 'music'
                if domain == 'sports':return 'match'
                if domain == 'stock':return 'stock'
                if domain == 'tvchannel':return 'tvchannel'
                if domain == 'website':return 'website'
            return None
        else:
            return None

        
    def classifyAllIntents(self,sample):
        """
        A classifier for 31 intents including chitchat
        :param sample: a dict indicates a query and its POS tag
        :return:a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_31.transform(text)
        test_wd = self.word_vectorizer_31.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        test_vec = self.ch2_.transform(test_vec)
        pred=self.clf_31.predict(test_vec)
        return pred.tolist()[0]


    def epgOrTvchannel(self,sample):
        """
        A classifier to label a instance with 'epg' or 'tvchannel'
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_tv.transform(text)
        test_wd = self.word_vectorizer_tv.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        pred=self.clf_tv.predict(test_vec)
        return pred.tolist()[0]


    def videoOrCinemas(self,sample):
        """
        A classifier to label a instance with 'video' or 'cinemas'
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_movie.transform(text)
        test_wd = self.word_vectorizer_movie.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        pred=self.clf_movie.predict(test_vec)
        return pred.tolist()[0]


    def websiteOrApp(self,sample):
        """
        A classifier to label a instance with 'website' or 'app'
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_internet.transform(text)
        test_wd = self.word_vectorizer_internet.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        pred=self.clf_internet.predict(test_vec)
        return pred.tolist()[0]


    def videoOrMusic(self,sample):
        """
        A classifier to label a instance with 'video' or 'music'
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_star.transform(text)
        test_wd = self.word_vectorizer_star.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        pred=self.clf_star.predict(test_vec)
        return pred.tolist()[0]


    def videoOrEpg(self,sample):
        """
        A classifier to label a instance with 'epg' or 'video'
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        raw_query=sample['query']
        text = [''.join([w for w in jieba.cut(raw_query)])]
        test_ch = self.char_vectorizer_video.transform(text)
        test_wd = self.word_vectorizer_video.transform(text)
        test_vec=hstack([test_ch, test_wd])
        test_vec=csr_matrix(test_vec)
        pred=self.clf_video.predict(test_vec)
        return pred.tolist()[0]


    def pipeline(self,sample,use_pse=True,use_retrieval=False):
        """
        A pipeline to label a instance with one of 31 possible intents
        :param sample: a dict indicates a query and its POS tag
        :return: a string indicates one certain intent
        """
        if use_pse:
            ps_res=prettySureExpression(sample['query'],self.ac)

            if len(list(set([_[1][0] for _ in ps_res])))==1:
                return ps_res[0][1][0]
        pm_res=self.pattern_match(sample)

        if pm_res=='TV_stuff':
            clf_res=self.classifyAllIntents(sample)# a ML classifier to label 31 intentions
            if clf_res in ['epg','tvchannel']:
                return clf_res
            else:
                return self.epgOrTvchannel(sample)#a ML classifier to label epg or tvchannel 

        elif pm_res=='Movie_stuff':
            clf_res=self.classifyAllIntents(sample)# a ML classifier to label 31 intentions
            if clf_res in ['video','cinemas']:
                return clf_res
            else:
                return self.videoOrCinemas(sample)

        elif pm_res=='Internet_stuff':
            clf_res=self.classifyAllIntents(sample)# a ML classifier to label 31 intentions
            if clf_res in ['website','app']:
                return clf_res
            else:
                return self.websiteOrApp(sample)

        elif pm_res=='Star_stuff':
            clf_res=self.classifyAllIntents(sample)# a ML classifier to label 31 intentions
            if clf_res in ['video','music']:
                return clf_res
            else:
                return self.videoOrMusic(sample)

        elif pm_res=='Video_stuff':
            clf_res=self.classifyAllIntents(sample)# a ML classifier to label 31 intentions
            if clf_res in ['video','epg']:
                return clf_res
            else:
                return self.videoOrEpg(sample)

        elif pm_res==None:

            if use_retrieval:
                ret_res=self.retrieval(sample,self.ac)
                if ret_res==None:
                    return self.classifyAllIntents(sample)# no pattern matched, so that classify it using ML
                else:
                    return ret_res
            else:
                return self.classifyAllIntents(sample)
        else:
            return pm_res


    def preprocess(self,raw_query):
        """
        To segment a raw user query into words and POS-tags it
        :param raw_query: a string generated by a user
        :return: a dict indicate the segmented query ,raw query and POS-tags
        """
        tmp = pseg.cut(raw_query)
        words=[]
        pos=[]
        for word, flag in tmp:
            words.append(word)
            pos.append(flag)
        inst={}
        inst['tag']=pos
        inst['word']=words
        del words
        del pos
        inst['query']=raw_query
        return inst

    def close(self):
        """
        To release relevant models
        """
        self.postagger.release()  # 释放模型
        self.segmentor.release()  # 释放模型
        self.labeller.release()  # 释放模型
        self.parser.release()  # 释放模型   
        del self.ac
        gc.collect()