#  -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by W. Y. Shen on 2018/7/16
"""
import sys
sys.path.append("..")
from config import configs
import ahocorasick
import os


common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,  
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}  
common_used_numerals = {}  

for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]  
  
  
def chinese2digits(uchars_chinese):  
    total = 0  
    r = 1  # 表示单位：个十百千...  
    for i in range(len(uchars_chinese) - 1, -1, -1):  
        val = common_used_numerals.get(uchars_chinese[i])  
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类  
            if val > r:  
                r = val  
                total = total + val  
            else:  
                r = r * val  
                # total =total + r * x  
        elif val >= 10:  
            if val > r:  
                r = val  
            else:  
                r = r * val  
        else:  
            total = total + r * val  
    return total  
  
  
num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',  
                        '十']  
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']  
  

def changeChineseNumToArab(oriStr):  
    lenStr = len(oriStr);  
    aProStr = ''  
    if lenStr == 0:  
        return aProStr;  
  
    hasNumStart = False;  
    numberStr = ''  
    for idx in range(lenStr):  
        if oriStr[idx] in num_str_start_symbol:  
            if not hasNumStart:  
                hasNumStart = True;  
  
            numberStr += oriStr[idx]  
        else:  
            if hasNumStart:  
                if oriStr[idx] in more_num_str_symbol:  
                    numberStr += oriStr[idx]  
                    continue  
                else:  
                    numResult = str(chinese2digits(numberStr))  
                    numberStr = ''  
                    hasNumStart = False;  
                    aProStr += numResult  
  
            aProStr += oriStr[idx]  
            pass  
  
    if len(numberStr) > 0:  
        resultNum = chinese2digits(numberStr)  
        aProStr += str(resultNum)  
  
    return aProStr  


def aggregate_domains(domains):
    '''
    将通过领域词匹配得到的涉及领域组成一个列表
    :param domains: 领域词匹配结果，一个元组列表
    :return: 涉及到的列表,[(domain1,[domain1_word1,domain1_word2]),(),...]
    '''
    related_domain=[]
    for hit in domains:
        related_domain+=hit[1][0].split(',')
    related_domain=list(set(related_domain))

    # print(_array)
    typeI=[]#将明确类（I类单独取出排序后放置在II类前面）
    typeII=[]
    for each in related_domain:
        if '_' in each:
            typeI.append(each)
        else:
            typeII.append(each)
    if len(typeI)==1 and len(typeII)==1:
        domains_words_tuple_list_sorted=[]
        for jj in typeI+typeII:
            domains_words_tuple_list_sorted.append((jj, [_[1][1] for _ in domains if jj in _[1][0]]))
        return domains_words_tuple_list_sorted
    elif len(typeI)==1 and len(typeII)>1:
        typeII_cnt = {}
        for each in typeII:
            cnt = 0
            for j in domains:
                if each in j[1][0].split(','):
                    cnt += 1
            typeII_cnt[each] = cnt
        typeII=[intent[0] for intent in sorted(typeII_cnt.items(), key=lambda x: x[1],reverse=True)]
        new_typeII = []

        for idx in range(1, len(typeII)):
            if typeII_cnt[typeII[idx]] < typeII_cnt[typeII[idx - 1]]:
                new_typeII.append(typeII[idx - 1])
            else:
                break
        domains_words_tuple_list_sorted = []
        for jj in typeI + new_typeII:
            domains_words_tuple_list_sorted.append((jj, [_[1][1] for _ in domains if jj in _[1][0].split(',')]))
        return domains_words_tuple_list_sorted

    elif len(typeI)==1 and len(typeII)<1:
        return [(typeI[0],[_[1][1] for _ in domains if _[1][0] == typeI[0]])]
    elif len(typeII)==1 and (len(typeI)==0 or len(typeI)>1):
        return [(typeII[0], [_[1][1] for _ in domains if _[1][0] == typeII[0]])]
    elif len(typeII)>1 and len(typeI)==0:
        typeII_cnt = {}
        for each in typeII:
            cnt = 0
            for j in domains:
                if each in j[1][0].split(','):
                    cnt += 1
            typeII_cnt[each] = cnt
        typeII_= [x[0] for x in sorted(typeII_cnt.items(), key=lambda x: x[1],reverse = True)]
        # print(typeII_)

        new_typeII=[]

        for idx in range(1,len(typeII_)):
            if typeII_cnt[typeII_[idx]]<typeII_cnt[typeII_[idx-1]]:
                new_typeII.append(typeII_[idx-1])
            else:break
        
        domains_words_tuple_list_sorted = []
        # for jj in new_typeII:
        for jj in typeII_:
            domains_words_tuple_list_sorted.append((jj, [_[1][1] for _ in domains if jj in _[1][0]]))
        return domains_words_tuple_list_sorted
    else:
        return None



def getSRL(query,segmentor,postagger,parser,labeller):
    words = list(segmentor.segment(query))

    postags = postagger.postag(words)  # 词性标注
    
    arcs = parser.parse(words, postags)
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    v_ne_list=[]
    for role in roles:
        for arg in role.arguments:
            v_ne_list.append((words[role.index], "".join(words[arg.range.start:arg.range.end+1]) ))
    
    v_ne_list=sorted(v_ne_list,key = lambda x:len(x[1]),reverse = True)
    return v_ne_list


def get_rule(query,ac_automaton):
    """
    使用规则词典匹配出可能的意图
    :param query: 用户utterance
    :return: 匹配结果
    """

    # query=changeChineseNumToArab(query.lower())
    query=query.lower()
    match_res = []
    for item in ac_automaton.rules.iter(query):
        match_res.append(item)
    return match_res


def get_predicate(query,ac_automaton):
    """
    使用谓词词典匹配出可能的意图
    :param query: 用户utterance
    :return: 匹配结果
    """
    query=changeChineseNumToArab(query.lower())
    match_res = []
    for item in ac_automaton.predicates.iter(query):
        match_res.append(item)
    return match_res


def doRetrieval(query,ac_automaton):
    """
    检索专有名词
    :param query: 用户utterance
    :param ac_automaton: ac自动机实例
    :return: 匹配结果
    """
    query=changeChineseNumToArab(query.lower())
    match_res = []
    for item in ac_automaton.proper_noun_all.iter(query):
        match_res.append(item)
    return match_res


def get_nouns(query,domain,ac_automaton):
    """
        使用词典匹配出可能的专有名词
        :param query: 用户utterance
        :param domain: 带查找的词汇所在的领域
        :param ac_automaton: ac自动机实例
        :return: 匹配结果
    """

    query=changeChineseNumToArab(query.lower())
    match_res = []
    for item in ac_automaton.domain_noun[domain].iter(query):
        match_res.append(item)
    return match_res


def prettySureExpression(query,ac_automaton):
    """
    十分确定意图的表达
    :param query: 用户utterance
    :param ac_automaton: ac自动机实例
    :return: 匹配结果
    """
    query=changeChineseNumToArab(query.lower())
    match_res = []
    for item in ac_automaton.pretty_sure.iter(query):
        match_res.append(item)
    return match_res


class ACAutomatons:
    def __init__(self):
        print('Initializing AC Automatons ... ')
        self.predicates = self.getACAutomaton('predicates')
        self.rules = self.getACAutomaton('rules')
        self.proper_noun_all = self.getACAutomaton('proper_noun_all')
        self.pretty_sure=self.getACAutomaton('pretty_sure')
        self.domain_noun=self.getDomainsACAutomaton()
        print('AC Automaton Ready')

    def getACAutomaton(self,type):
        type2path={"predicates":configs.PREDICATES,
                    "rules":configs.RULES,
                    "proper_noun_all":configs.PROPER_NOUN_ALL,
                    "pretty_sure":configs.PRETTY_SURE}
        path=type2path[type]
        A = ahocorasick.Automaton()
        with open(path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                A.add_word(line.split('\t')[0], (line.split('\t')[1].strip(), line.split('\t')[0]))
        A.make_automaton()
        return A
        
    def getDomainsACAutomaton(self):
        files=[file for file in os.listdir(configs.PROPER_NOUN_PARENT) if os.path.splitext(file)[-1]=='.tsv']
        domain2ac={}
        
        for domain in files:
            try:
                A = ahocorasick.Automaton()
                with open(os.path.join(configs.PROPER_NOUN_PARENT,domain), 'r') as f:
                    for idx, line in enumerate(f.readlines()):
                        A.add_word(line.split('\t')[0], (line.split('\t')[1].strip(), line.split('\t')[0]))
                A.make_automaton()
                domain2ac[domain.replace('.tsv','')]=A
            except:
                print(domain)
        return domain2ac
