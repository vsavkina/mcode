
# coding: utf-8

# In[2]:

import re
import csv
import spacy, ner, treetaggerwrapper
from _pickle import UnpicklingError
import langid, os
from rusynt import *
import keras
import treetaggerwrapper
import tkinter as tk
from tkinter.filedialog import *
import tkinter.ttk as ttk
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import messagebox as msg
import zipfile, os
from lxml import etree
from collections import OrderedDict
import zipfile, os
from lxml import etree
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


# In[3]:

from gensim.models import word2vec
from nltk.cluster import KMeansClusterer
from itertools import permutations, groupby, takewhile
from sklearn.feature_extraction.text import CountVectorizer
from numpy import concatenate,array
from itertools import combinations
import nltk
from math import inf
from string import ascii_lowercase
from collections import namedtuple, Counter


# In[4]:

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[5]:

import networkx
import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import random


# In[6]:

from conllu import parse, parse_tree, print_tree


# In[8]:

extractor = ner.Extractor()


# In[9]:


ru_morph = treetaggerwrapper.TreeTagger(TAGLANG='ru', TAGDIR = r'C:\Users\user\Desktop\СРЯ\TreeTagger')
extractor = ner.Extractor()
path_to_ger_model = r"C:\Users\user\AppData\Local\conda\conda\envs\my_root\lib\site-packages\xx"
nnlp = spacy.load("en")
nlp_de = spacy.load(path_to_ger_model)
nnlp_de = spacy.load('de')


# In[10]:

text_ru_split, text_en_split, text_de_split = [],[],[]


# In[11]:

def find_sub_list(sl,l):
    """auxiliary function for locating named entities in text """
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append([ind,ind+sll-1])

    return results


# In[12]:

lmtzr = nltk.WordNetLemmatizer()
split_capitals = lambda string: [a for a in re.split(r'([А-ЯA-Z][а-яa-z]*)', string) if a] #split a string by capital letters


# In[13]:

langid.set_languages(['de','en','ru'])


# In[14]:


repl_dash = lambda s: s.replace('-', '')
ne_en = []
def process_en(text, text_en_split, ne_en):
    sentences = sent_tokenize(text)
    tagged_text = [[ (repl_dash(item[0]), item[1], repl_dash(lmtzr.lemmatize(item[0], pos = item[1][0].lower())))
                    if item[1][0].lower() in ['a','n','v']
                    else (repl_dash(item[0]), item[1], repl_dash(lmtzr.lemmatize(item[0])))
                    for item in pos_tag(word_tokenize(s))] for s in sentences]
    chunks = [ne_chunk(ttext).subtrees(filter=lambda subtree: subtree.label() == 'PERSON')
                  for ttext in tagged_text]
    entities_per_text = [[x[0] for x in st.leaves()] for stree in chunks for st in stree]
    for sent_i, sent in enumerate(tagged_text):
        for entity_found in entities_per_text:
            indices_of_entity = find_sub_list(entity_found, [t[0] for t in sent])
            entity_found, entity_lemma = ''.join(entity_found), ''.join(lmtzr.lemmatize(w) for w in entity_found)
            for index_pairs in indices_of_entity:
                tagged_text[sent_i] = tagged_text[sent_i][:index_pairs[0]] + [ (entity_found,'NE', entity_lemma) ] + tagged_text[sent_i][index_pairs[1] + 1:]
            if entity_lemma not in ne_en:
                ne_en.append(entity_lemma) 
    text_en_split.append(tagged_text)


# In[15]:


ne_ru = []
def process_rus(text, text_ru_split, ne_ru):
    if not text.strip():
        return ''
    tagged_text = [[ (repl_dash(tag.word), tag.pos, repl_dash(tag.lemma)) if hasattr(tag, 'word') else tuple(tag.what.split())
                   for tag in treetaggerwrapper.make_tags(ru_morph.tag_text(txt)) ]
                   for txt in sent_tokenize(text)]
    for m in extractor(text):
        if m.type == 'PER':
            entity_found = [tok.text for tok in m.tokens]
            
            if False in [w[0].isupper() for w in entity_found]:
                continue
            for s in range(len(tagged_text)):
                indices_of_entity = find_sub_list(entity_found, [t[0] for t in tagged_text[s]])
                ent_found, ent_lemma = ''.join(entity_found), ''.join(tok.lemma[0].upper() + tok.lemma[1:] for tok in treetaggerwrapper.make_tags(ru_morph.tag_text(entity_found)))
                for index_pairs in indices_of_entity:
                    tagged_text[s] = tagged_text[s][:index_pairs[0]] + [ (ent_found,'NE', ent_lemma) ] + tagged_text[s][index_pairs[1] + 1:] 
            if ent_lemma not in ne_ru:
                ne_ru.append(ent_lemma)
    text_ru_split.append(tagged_text)   


# In[16]:

ne_de = []
def process_de(text, text_de_split, ne_de):
    if not text.strip():
        return ''
    doc = nlp_de(text)
    tagged_text = [word_tokenize(sent) for sent in sent_tokenize(text)]
    for s in range(len(tagged_text)):
        morph_features = [(token.text, token.lemma_, token.pos_) for token in nnlp_de(' '.join(tagged_text[s]))]
        tagged_text[s] = morph_features
    
    for entity_found in doc.ents:
        if entity_found.text.strip() and entity_found.label_ == 'PER':
            entity_found, entity_lemma = entity_found.text, entity_found.lemma_
            for s in range(len(tagged_text)):
                indices_of_entity = find_sub_list(entity_found.split(), [t[0] for t in tagged_text[s]])
                entity_found = entity_found.replace(' ', '')
                entity_lemma = entity_lemma[0].upper() + entity_lemma[1:].replace(' ', '') 
                for index_pairs in indices_of_entity:
                    tagged_text[s] = tagged_text[s][:index_pairs[0]] + [ (entity_found,'NE', entity_lemma) ] + tagged_text[s][index_pairs[1] + 1:]
            if entity_lemma not in ne_de:
                ne_de.append(entity_lemma)
    text_de_split.append(tagged_text)


# In[17]:

def coreference(mod, model, ln, text_split, ne_clusts):
    lemmas = [w[2] for sent in text_split for w in sent if sent]
    wordforms = [w[0] for sent in text_split for w in sent if sent]
    text = ' '.join(wordforms)
    res = {}
    last_mentioned = []
    if ln == 'ru':
        n_gender =  dict(zip(ne_clusts, [word['xpostag'].split()[1] for word in parse_rus_texts(' '.join(split_capitals(w)[0] for w in ne_clusts), False)[0][:-1]]))
        parsed_text = model(text, False)
        for branch in parsed_text:
            for token in branch:
                if len(token['form'].split()) > 1 or token['form'] not in wordforms: continue
                if token['upostag'] == 'SPRO' and token['lemma'].startswith('ОН'):
                    if token['xpostag'].split()[1] == 'sg':
                        try:
                            res[(token['form'], token['head'], token['id'])] = res[next(actor['form'] for actor in last_mentioned[::-1] if n_gender[lemmas[wordforms.index(actor['form'])]] == token['xpostag'].split()[3] )]
                        except StopIteration:
                            pass
                    elif len(last_mentioned) > 1:
                        res[(token['form'], token['head'], token['id'])] = (res[last_mentioned[-2]['form']], res[last_mentioned[-1]['form']])
                elif lemmas[wordforms.index(token['form'])] in ne_clusts:
                    res[token['form']] = ne_clusts[lemmas[wordforms.index(token['form'])]]
                    last_mentioned.append(token)
                    if len(last_mentioned) > 2 and last_mentioned[0]['xpostag'].split()[1] == token['xpostag'].split()[1]:
                        last_mentioned.pop(0)
                elif token['upostag'] == 'S' and token['xpostag'].split()[2] == 'anim' and token['form'].upper() != token['form']:
                    plural = False if token['xpostag'].split()[-1] == 'sg' else True
                    try:
                        res[(token['form'], token['head'], token['id'])] = get_cluster_num(mod, lemmas[wordforms.index(token['form'])].lower(), {k.lower():v for k,v in ne_clusts.items() if k in lemmas and (token['xpostag'].split()[1] == 'm' or n_gender[k] == token['xpostag'].split()[1] ) }, plural)
                    except IndexError:
                        pass
                
                
    else:
        parsed_text = model(text)
        for tok_index, token in enumerate(parsed_text):
            if token.text not in wordforms: continue
            if last_mentioned and token.pos_ == 'PRON' and len(token.text) > 1 :
                if token.text.lower()[:2] not in ('wi', 'eu', 'th', 'ih', 'it', 'es', 'yo'):
                    res[(token.text, tok_index)] = res[ last_mentioned[-1] ]
                elif token.text.lower()[:2] not in ('it','es', 'yo') and len(last_mentioned) > 1:
                    res[(token.text, tok_index)] = (res[ last_mentioned[-2] ], res[ last_mentioned[-1] ])
            elif lemmas[wordforms.index(token.text)] in ne_clusts:
                res[(token.text, tok_index)] = ne_clusts[lemmas[wordforms.index(token.text)]]
                last_mentioned.append((token.text, tok_index))
    return res
            


# In[18]:

def get_cluster_num(model, tok, clusters_dict, plural = False):
    similarity = sorted(clusters_dict, key = lambda item: model.wv.similarity(tok, item))
    if plural:
        return (clusters_dict[similarity[-2]], clusters_dict[similarity[-1]])
    return clusters_dict[similarity[-1]]


# In[19]:

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def clust_ne(l):
    l = sorted(l, key= lambda i: len(i))
    visited = []

    clsts, res = [], []
    for item in l:
        if item in visited: continue
        tmp = []
        for sec_item in l:
            if sec_item not in visited and split_capitals(sec_item)[-1] == split_capitals(item)[-1]:
                tmp.append(sec_item)
                visited.append(sec_item)
        clsts.append(tmp)
        visited.append(item)
    
    visited = []
    for clst in clsts:
        for item in sorted(clst)[::-1]:
            if item in visited: continue
            tmp = []
            for sec_item in sorted(clst)[::-1]:
                if sec_item not in visited and (split_capitals(sec_item)[0] == split_capitals(item)[0] or levenshteinDistance(sec_item.lower(),item.lower()) < 3):
                    tmp.append(sec_item)
                    visited.append(sec_item)
            res.append(tmp)
            visited.append(item)

    l_clsts = []
    for lon_clsts in filter(lambda clst: len(clst) == 1, res):
        merged = False
        for clst in res:
            if len([x for x in clst if lon_clsts[0] in x and lon_clsts[0] != x]) == len(clst):
                clst.append(lon_clsts[0])
                merged = True
        if not merged: l_clsts.append(lon_clsts)


        

    clusters = {}
    for cl_i, clstr in enumerate( [list(set(clst)) for clst in res if len(clst) > 1] + l_clsts ):
        for item in clstr:
            if item not in clusters:
                clusters[item] = [cl_i]
            else:
                clusters[item].append(cl_i)
    return(clusters)


# In[20]:

def parse_rus_texts(text, return_tree = True):
    """Returns a dependency tree for russian texts"""
    tmp_inp = 'inp.txt'
    with open(tmp_inp, 'w', encoding = 'utf8') as f:
        f.write(text)
    text_tree = parse_sent('inp.txt','out.conll', return_tree)
    os.remove(tmp_inp)
    return text_tree


# In[21]:

def find_obj(len_prev_sents, tok_to_ind, predicate, actors):
    res = [predicate]
    for ch in predicate.children:
        if ch.dep_ != 'sb':
            if ch.dep_ == 'dobj' or ch.pos_ == 'NOUN' or ch.dep_ == 'prep' or ch.pos_ == 'ADP':
                res.extend(dfs_objs(ch, [], verb_objects, None))
    return [ch for ch in res if (ch.text, abs_index(len_prev_sents, tok_to_ind, ch.text)) not in actors]
def dfs_objs(node, visited, rule, exception_subj, head_of_tree = None):
    has_subjs = [ch for ch in node.children if ch != exception_subj and ('subj' in ch.dep_ or ch.dep_ == 'sb')]
    if node not in visited and not has_subjs and node != exception_subj:
        visited.append(node)
        chldr = rule(node)
        for n in chldr:
            if not (node == head_of_tree and (n.dep_ == 'conj' or n.dep_ == 'cj' )):
                dfs_objs(n, visited, rule, exception_subj, head_of_tree)
    return visited
def find_related_ne(len_prev_sents, tok_to_ind, subj, actors):
    
    rel_ne = None
            
    if subj.text not in actors:

        try:

            rel_ne = next(ch for ch in subj.subtree if (ch.text, abs_index(len_prev_sents, tok_to_ind, ch.text)) in actors and 'cj' != ch.dep_ != 'conj'  )

        except StopIteration:

            pass
    
    return (subj, rel_ne)


# In[22]:

def add_obj_images(res_en, dic, a_o, lps, tti, subjname = None):
    
    for obj in a_o:
        
        if type(obj[0]) != list:
            
            continue
                        
        value = [(obj[0], (subjname.text, abs_index(lps, tti, subjname.text))  ), *chars(obj[1])] if subjname else [[], *chars(obj[1])]

        if len(obj) > 1 and (obj[1].text, abs_index(lps, tti, obj[1].text)) in res_en: dic.setdefault((obj[1].text, abs_index(lps, tti, obj[1].text)), []).extend(value)  



# In[23]:


#rules

find_conj = lambda o: [ch for ch in o.children if ch.dep_ == 'conj' or ch.dep_ == 'cj' ]

all_chldr = lambda node: node.children

verb_objects = lambda o: [ch for ch in o.children if 'obj' in ch.dep_ or ch.pos_ == 'NOUN' or ch.dep_ == 'prep' or ch.pos_ == 'ADP']

auxiliary_rule = lambda predicate: [ch for ch in predicate.children if 'aux' in ch.dep_ or 'AUX' in ch.pos_]
        
infs_rule = lambda predicate: [ch for ch in predicate.children if ch.dep_ in ('xcomp', 'acomp', 'advcl', 'app', 'attr', 'neg', 'ng', 'pd') and not [c for c in ch.children if 'subj' in c.dep_] 
                               or ch.tag_.startswith('VV') ]
def chars(person):
    
    res = []
    
    res_nk = []
    
    for ch in person.children:
    
        if ch.dep_ in ('app', 'compound', 'appos'):
            
            res.append(tuple(dfs_objs(ch, [], all_chldr, None)))
            
        elif ch.dep_ == 'nk':
            
            res_nk.extend(dfs_objs(ch, [], all_chldr, None))
    
    if res_nk:
        res.append(tuple(res_nk))
    
    return res
abs_index = lambda lps, ti, tok: lps + ti[tok]    


def get_relations_from(txt, res_en, ln, lemmas):
    
    sub_marker = 'subj' if ln == 'en' else 'sb'
    conj_marker = 'conj' if ln == 'en' else 'cj'
    cc_marker = 'cc' if ln == 'en' else 'cd' 
    
    subjs = {}
    len_prev_sents = 0
    txt = [' '.join([t[0] for t in sent]) for sent in txt]
    
    for sentence_index, sent in enumerate(txt):
        
        
        t = nnlp(sent) if ln == 'en' else nnlp_de(sent)
        tok_to_index = { token.text: index for index, token in enumerate(t)}
   
        sent_subjs = [tok for tok in t if sub_marker in tok.dep_]
    
        subj_heads = [ (tok,hd) for tok in sent_subjs for hd in dfs_objs(tok.head, [], find_conj, tok) ]
        
        for subj, head in subj_heads:

            subj_conjs = [tok for tok in t if conj_marker in tok.dep_ and (tok.text, abs_index(len_prev_sents, tok_to_index, tok.text))  in res_en and subj.is_ancestor(tok)]

            dpds = dfs_objs(head, [], all_chldr, subj, head)
            
            dependants = [dp for dp in dpds if (dp.text, abs_index(len_prev_sents, tok_to_index, dp.text)) in res_en and dp != subj and dp not in subj_conjs]
            
            way_to_obj =  [tuple(takewhile(lambda an: an != head, dep.ancestors))[::-1] for dep in 
                           [ dp if dp.dep_ != conj_marker else next(an for an in dp.ancestors if an.dep != conj_marker and an.dep != cc_marker) for dp in dependants ] ]

            #get auxiliary verbs

            auxs_head = auxiliary_rule(head)

            #get infinitives and their children iteratively

            infs = dfs_objs(head, [], infs_rule, subj, head) 

            #get conjuncts of the infinitives and their dependant infinitives

            infs.extend([verb for token in infs for tok in token.children for verb in dfs_objs(tok, [], infs_rule, subj) if tok.dep_ == conj_marker])

            infs = [it for item in infs for it in find_obj(len_prev_sents, tok_to_index, item, res_en) if it in dpds and it not in [o for objl in way_to_obj for o in objl] ]

            pred = auxs_head + infs

            if dependants:

                a_o = list( ( sorted(tuple([p for p in (*pred, *p) if tok_to_index[p.text] >= tok_to_index[head.text] or p.dep_ in ('neg', 'aux')]), key = lambda w: tok_to_index[w.text]) , a )  for p,a in zip(way_to_obj, dependants))[:5]
                
            else:

                if subj_conjs:

                    dependants = subj_conjs

                    subj_conjs = []

                    a_o = [(sorted(tuple([p for p in pred if tok_to_index[p.text] >= tok_to_index[head.text] or p.dep_ in ('neg', 'aux')]),key = lambda w: tok_to_index[w.text]) , dep) for dep in dependants][:5]

                else:

                    a_o = [sorted(tuple([p for p in pred if tok_to_index[p.text] >= tok_to_index[head.text] or p.dep_ in ('neg', 'aux')]),key = lambda w: tok_to_index[w.text])][:5]


            for conj in [subj] + subj_conjs:

                subjname, rel_ne = find_related_ne(len_prev_sents, tok_to_index, conj, res_en)

                if (subjname.text, abs_index(len_prev_sents, tok_to_index, subjname.text)) in res_en or rel_ne:
                    
                    sname, char = (subjname.text, chars(subjname)) if not rel_ne else (rel_ne.text, chars(subjname) + [(subjname,)])
                    
                    for relation in a_o:
                        
                        if relation and type(relation[0]) == list:
                            
                            subjs.setdefault((sname, abs_index(len_prev_sents, tok_to_index, sname)),[]).append( (relation[0], (relation[1].text, abs_index(len_prev_sents, tok_to_index, relation[1].text) ), lemmas[head.text] )) 

                        elif relation:
                            
                            subjs.setdefault((sname, abs_index(len_prev_sents, tok_to_index, sname)),[]).append( (relation, lemmas[head.text] ) )

                    subjs.setdefault((sname, abs_index(len_prev_sents, tok_to_index, sname)),[]).extend(char)
                    
                    add_obj_images(res_en, subjs, a_o, len_prev_sents, tok_to_index)

                else:
                    
                    add_obj_images(res_en, subjs, a_o, len_prev_sents, tok_to_index, subjname)
                    
        len_prev_sents += len(t)

    for subj in subjs:
        for relation in subjs[subj]:
            if len(relation) == 3 and type(relation[0]) == list:
                yield [subj,  relation[1], relation[0], relation[2]]
            elif len(relation) == 2 and type(relation[1]) == tuple:
                yield [subj,  relation[1], relation[0], lemmas[head.text]]
            elif len(relation) == 2 and type(relation[0]) == list:
                yield [subj, None, relation[0], relation[1]]
            elif relation and relation[0].text != subj[0]:
                yield [subj, None, relation, lemmas[relation[0].text]]
    


# In[25]:

def bfs(start):
    visited, queue, res = [], [start], []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            queue.extend([item for item in vertex.children if item not in visited])
            if vertex.data['upostag'] in ('V', 'N', 'A', 'S'):
                break
    visited = [child for child in visited if child.data['upostag'] in ('V', 'N', 'A', 'S') ]
    if visited:return visited[-1]


# In[26]:

subjs = lambda x: (ch for ch in x if ch.data['deprel'] in ['предик', "дат-субъект", "опред"])
def is_pred(predic, pr, x, actors):
    pred_children = pr if type(predic) == OrderedDict else predic.children
    if predic == x or (retrieve_value(x, 'deprel') == 'опред' and '—' in [retrieve_value(ch , 'lemma') for ch in pred_children]):
        return True
    if retrieve_value(predic, 'lemma') in ('—', 'БЫТЬ') and retrieve_value(x, 'deprel') in ('1-компл', 'присвяз') and retrieve_value(x, 'head') == retrieve_value(predic, 'id'):
        return True
    if retrieve_value(x, 'deprel') == 'аппоз' and '—' in [retrieve_value(ch , 'lemma') for ch in x.children if retrieve_value(ch , 'deprel') == 'опред']:
        return True
def objs(x, actors, depth = None):
    if type(x)!=OrderedDict:
        return( x.data['form'] in actors or (x.data['form'], x.data['head'], x.data['id']) in actors or [w for w in x.children if w.data['form'] in actors and "аппоз" in w.data['deprel']])
    return( x['form'] in actors or (x['form'], x['head'], x['id']) in actors)
actor_in_appos = lambda y, actors: next(x for x in y if retrieve_value(x, 'form') in actors and 'аппоз' in retrieve_value(x, 'deprel') and retrieve_value(x, 'head') == retrieve_value(y[0], 'id'))    
def predicates(x, actors, depth):
    return(x.data['form'] not in actors and (x.data['form'], x.data['head'], x.data['id']) not in actors and x.data['deprel'] in ('1-компл', '2-компл', '3-компл', '4-компл', "пасс-анал", "аналит", "присвяз", "предл", "сравнит", "сравн-союзн", "количест", 'огранич', 'опред', 'вспом') and not (True in [ch.data['deprel'] in ("пасс-анал", "аналит", '1-компл', "присвяз", "предл", "сравнит", "сравн-союзн", "количест", 'огранич', 'опред', 'вспом') for ch in x.children if depth > 0]))
attributes = lambda x, actors = None, depth = None: x.data['deprel'] in ('1-компл', '2-компл', '3-компл', '4-компл', 'обст', 'атриб', 'квазиагент', 'аппоз', "сравн-союзн", 'об-аппоз', "присвяз", "предл", 'опред') and not (True in [ch.data['deprel'] in ('1-компл', '2-компл', 'квазиагент', '3-компл', '4-компл', "обст" ,'атриб', 'аппоз', "сравн-союзн", 'об-аппоз', "присвяз", "предл", 'опред') for ch in x.children])
sort_by_id = lambda attribs, actor = None, length = None, start_from_head = True: sorted( (x for x in attribs if x != actor and (True if not start_from_head else retrieve_value(x, 'id') >= retrieve_value(attribs[0], 'id') ) ), key=lambda i: i.data['id'] if type(i)!=OrderedDict else i['id'])[:length]
def retrieve_value(act, key): 
    if act == None: return None
    return(act.data[key] if type(act)!=OrderedDict else act[key])
def get_relations_from_ru(mtext, actors, lemmas):
    mtext = ' '.join([t[0] for sent in mtext for t in sent])
    for tree in parse_rus_texts(mtext):

        root, children = [tree[0]], tree[1]
        if root[0]['upostag'] not in ('V', 'N', 'A', 'S'):
            root = [i for i in [bfs(child) for child in children] if i]
            g = [bfs_conj(i, True) for i in root]
            tmp = []        
            for index,it in enumerate(g):
                for key,value in it.items():
                    tmp.append([root[index], *value]) if key == root[index].data['id'] else tmp.append(value)
            root = tmp
        else:
            if children:
                g = bfs_conj(children, True)
                g[root[0]['id']] = root + g[root[0]['id']]
                root = list(g.values())
            else:
                root = [root]

        all_objs = []
        for index, group in enumerate(root):
            try:
                subj = subjs(children) if type(group[0]) == OrderedDict else subjs(group[0].children)
                all_sb = [item for sub in subj for item in [sub, *bfs_conj(sub, True)[sub.data['id']]] ]
                
            except StopIteration:
                all_sb = []
            obj = []
            for pred in group:
                pr = children if type(pred) == OrderedDict else pred
                ss = merge_lists(*bfs_paths(actors, pred, pr, None, rule = predicates, exception =  all_sb + [v for gr in root for v in gr] if all_sb else [v for gr in root for v in gr]))
                if not ss:
                    ss = [pred]
                o = [[item[-1],  *bfs_conj(item[-1], True)[item[-1].data['id']]] for item in bfs_paths(actors, pred, pr, None, rule = objs, exception = all_sb + [v for gr in root for v in gr] if all_sb else [v for gr in root for v in gr]) ]
                if o:
                    ss.extend([word for ob in o for word in ob if is_pred(pred, pr, ob[0], actors) and retrieve_value(word, 'form') not in actors])
                ss.extend([item for item in all_sb if item.data['form'] not in actors and (item.data['form'], item.data['head'], item.data['id']) not in actors ])
                o = [word for word in all_sb if word not in ss] + [word for ob in o for word in ob if word not in ss]
                if retrieve_value(pred, 'form') in actors:
                    o.append(ss.pop(0))
                
                if o:
                    obj.append([o, ss])
                    
                
            all_objs.append(obj)
        for clause in range(len(root)):

            for nwsrmkrs, prdc in all_objs[clause]:
                
                attrs = [[tuple(attribs) for attribs in bfs_paths(actors, None, j, None, attributes, depth = 10)] if type(j) != OrderedDict else j for j in nwsrmkrs ]
                for sub in attrs:
                    if type(sub) != list:
                        continue
                    actor = None 
                    for item in sub:
                        if not item: continue
                        try:
                            actor = actor_in_appos(item, actors)
                            break
                        except StopIteration:
                            actor = item[0]

                    if actor:

                        nwsrmkrs[attrs.index(sub)] = actor

                        if objs(actor, actors) and actor not in prdc:

                            for attr in sub:
                                
                                if len(attr) < 2: continue

                                actor_value = retrieve_value(actor,'form')
                                
                                if actor_value not in actors: actor_value = (actor_value, retrieve_value(actor,'head'), retrieve_value(actor,'id'))
                                
                                yield [actor_value, None, [retrieve_value(c,'form') for c in sort_by_id(attr, actor, start_from_head = False)], lemmas[retrieve_value(attr[0],'form')]]
                                
                if not prdc:
                    continue
                combs = combinations(nwsrmkrs, 2) if len(nwsrmkrs) > 1 else [(nwsrmkrs[0],None)]
                for comb in combs:
                    rest = []
                    for actor in comb[0], comb[1]:
                        
                        actor_value = retrieve_value(actor,'form')
                                
                        if actor_value not in actors: actor_value = (actor_value, retrieve_value(actor,'head'), retrieve_value(actor,'id'))
                            
                        rest.append(actor_value)
                    try:
                        lms = lemmas[retrieve_value(prdc[0],'form')]
                    except KeyError:
                        lms = lemmas[retrieve_value(prdc[0],'form').split()[-1]]
                    rest.extend([[retrieve_value(c,'form') for c in sort_by_id(prdc, length = 5)], lms])
                    yield rest



# In[27]:

def bfs_conj(start, groupby_subj = False):
    visited = []
    if type(start) == list:
        queue = start[:]
        heads_ids = [start[0].data['head']]
    else:
        queue = start.children[:]
        heads_ids = [start.data['id']]
    while queue:
        vertex = queue.pop(0)
        res = []
        if vertex not in visited and vertex.data['head'] in heads_ids:
            tmp_chldr, depth = vertex.children[:], 4
            while tmp_chldr and depth:
                ch = tmp_chldr.pop(0)
                if ch.data['deprel'] == 'эксплет':
                    res.extend([child for child in ch.children if 'союзн' in child.data['deprel'] or 'сент' in child.data['deprel']])
                    break
                tmp_chldr.extend(ch.children)
                depth -= 1
            if vertex.data['upostag'] == 'CONJ':
                res.extend([child for child in vertex.children if 'союзн' in child.data['deprel'] or 'сент' in child.data['deprel']])
            elif vertex.data['deprel'] == 'сент-соч':
                res.append(vertex)
        if res:
            visited.extend(res)
            heads_ids.extend([child.data['id'] for child in res])
            queue.extend([item for child in res for item in child.children if item not in visited])

    if groupby_subj:
        res = {heads_ids[0]: []}
        res_groupby = groupby(visited, lambda item: item.data['id'] if [child for child in item.children if child.data['deprel'] in ['предик', "дат-субъект", "опред"]] else heads_ids[visited.index(item)])
        res.update({key:list(group) for key, group in res_groupby})
        return(res)
    return [ch.data['form'] for ch in visited]


# In[28]:

def bfs_paths(actors, pred, start, goal, rule = None,  exception = [], depth = 5):
    queue = [(pred, [pred])] if type(start) == list else [(start, [start])]
    while queue and depth:
        depth -= 1
        (vertex, path) = queue.pop(0)
        children = start if type(vertex) == OrderedDict else vertex.children
        items = [item for item in children if item not in path and item not in exception]
        for next in items:
            if next == goal or (rule and rule(next, actors, depth)):
                yield path + [next]
            else:
                queue.append((next, path + [next]))


# In[29]:

def merge_lists(*lists):
    res = []
    for l in lists:
        for item in l:
            if item not in res: res.append(item)
    return(res)


# In[ ]:



# In[ ]:




class Application(Tk):

    def __init__(self):
        
        self.w2v_models = {}

        Tk.__init__(self)

        self.title('')
        self.geometry('786x640')

        self.main_window = ttk.Frame(self, width=768, height = self.winfo_screenheight())
        self.main_window.pack(fill = BOTH)
        self.main_window.pack_propagate(0)

        self.canvas = Canvas(self.main_window, width = 200, borderwidth=0, background="#ffffff")
        
        self.corp_window = ttk.Frame(self.canvas, width = 200, style = 'CL.TFrame')

        self.vsb = Scrollbar(self.main_window, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="left", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.corp_window, anchor="nw", 
                                  tags="self.corp_window")

        self.corp_window.bind("<Configure>", self.onFrameConfigure)

        self.tabs = ttk.Notebook(self.main_window)
        self.tabs.pack(fill=BOTH, expand=TRUE)
        self.protocol("WM_DELETE_WINDOW", self.m_quit)

        self.menu_bar = Menu(self)
        self.config(menu=self.menu_bar)
        #configure styles
        self.style = ttk.Style()
        self.style.configure('CL.TFrame', background='white', border=6)
        self.style.map("Menu.TFrame",
            foreground=[('active', 'blue')],
            background=[('active', 'white')]
            )
        self.style.map("Corp.TFrame",
                  background=[('active', 'gray')])
        self.style.configure('WC.TFrame', background='gray')
        #menu options
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label='Open', command=self.open_file)
        self.file_menu.add_command(label='Open dir', command=self.open_dir)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Exit', command=self.m_quit)
        self.menu_bar.add_cascade(label='File',menu=self.file_menu)

        self.analyze = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Process texts',menu=self.analyze)
        self.analyze.add_command(label='Get images and relations', command=self.process_texts)

        self.tab_view = ttk.Frame(self.tabs, style='Menu.TFrame')
        self.tabs.add(self.tab_view, text='File View')
        text = Text(self.tab_view)

        #file view
        self.text_window = ttk.Frame(self.tab_view, width = self.winfo_screenwidth() - 200, height = self.winfo_screenheight(),
                                     style = 'CL.TFrame')
        self.text_window.pack()
        self.text_window.pack_propagate(0)
        
        self.text_field = Text(self.text_window)
        self.text_field.pack(fill = BOTH, expand = True)
        
        self.corpora = OrderedDict({'Main' : dict()}) #name of corpus : list of files
        self.active_corp = 'Main'
        self.name_to_button = {'Main': Button(self.corp_window, text = 'Main', command = lambda: self.make_active("Main"), background = 'white', relief = 'flat')}
        self.name_to_button['Main'].pack(anchor = W, padx = 15, pady = (15, 0))

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def m_quit(self): #попытка выйти из программы
        answer = msg.askyesno('', 'Are you sure to quit?')
        if answer == True:
            
            self.quit()
            self.destroy()
            exit()
            
    def readfromfile(self, path):
        if not path:
            return
        try:
            if path.endswith("docx"):
                with open(path,  'rb') as f:
                    zip = zipfile.ZipFile(f)
                    xml_content = zip.read("word/document.xml")
                    tree = etree.fromstring(xml_content)
                    res = ''.join(node.text for node in tree.iter(tag=etree.Element) if node.text)
                    if res.startswith('\ufeff'): res = res[1:]
            else:
                with open(path,  'r', encoding = 'utf-8') as f:
                    res = f.read()
                    if res.startswith('\ufeff'): res = res[1:]
            return res
        except UnicodeDecodeError as e:
            msg.showinfo("Error while loading the file", "File {} should have either .docx or .txt extension and be in UTF format.".format(path.split('/')[-1]))
     
    def open_file(self):
        op = askopenfilename()
        if op:
            self.update_filelist(op.split('/')[-1], self.readfromfile(op))

        #здесь будет функция открытия файлов
    def open_dir(self):
        op = askdirectory()
        if not op:
            return
        namedir = op.split('/')[-1]
        onlyfiles = [os.path.join(op, f) for f in os.listdir(op) if os.path.isfile(os.path.join(op, f)) and (f.endswith("docx") or f.endswith("txt"))]
        files = [ f.split('\\')[-1] for f in onlyfiles]
        name_to_cont = {}
        for file in range( len(onlyfiles) ):
            name_to_cont[files[file]] = self.readfromfile(onlyfiles[file])
        self.update_filelist(namedir, name_to_cont, False)
        
    def get_relations_graph(self, lang):
        texts, cl_to_name = self.proc_res[lang][0], self.proc_res[lang][2]
        G = networkx.Graph()
        cooccurences = Counter([tuple(t[:2]) for t in texts if type(t[1]) == int])
        for edge in cooccurences:
            G.add_edge(cl_to_name[edge[0]], cl_to_name[edge[1]], weight = cooccurences[edge] * 2, length = 1 / self.w2v_models[lang].wv.similarity(cl_to_name[edge[0]].lower(), cl_to_name[edge[1]].lower()))
        self.draw_graph_on_sep_window(G, networkx.draw_shell, with_labels=True, font_weight='bold')
        networkx.draw_shell(G, with_labels=True, font_weight='bold')
        
        plt.show()
        
    def get_precise_rels(self, central, other, lang):
        texts, cl_to_name = self.proc_res[lang][0], self.proc_res[lang][2]
        G = networkx.Graph()
        central_node_name = cl_to_name[central] if len(other) != 1 else ';'.join([cl_to_name[central],cl_to_name[other[0]]])
        G.add_node(central_node_name)
        fixedpos = {central_node_name: (0.5, 0.5)}
        cooccurences = []
        if not other:
            cooccurences.extend([t[:-1] for t in texts if (t[0] == central) ^ (t[1] == central) and t[1] != '-'])
        else:
            for actor in other:
                cooccurences.extend([t[:-1] for t in texts if not [x for x in (central, actor) if x not in (t[0],t[1])]])
        for edge in cooccurences:
            rel = ' '.join([e.text if type(e) != str else e for e in edge[2]])
            G.add_edge(central_node_name, rel)
            fixedpos.update({rel: (random.random(), random.random()) })
            if len(other) != 1:
                G.add_edge(rel, cl_to_name[edge[1]])  
        colors = ['g' if not index else 'r' for index,node in enumerate(G.nodes())]
        pos = networkx.spring_layout(G, fixed = fixedpos.keys(), pos = fixedpos)
        self.draw_graph_on_sep_window(G, networkx.draw_networkx, pos=pos, node_color = colors)
        networkx.draw_networkx(G, pos=pos, node_color = colors)
        plt.show()
            
    def get_personal_graph(self, heads, lang, person):
        G = networkx.Graph()
        G.add_node(person)
        if len(heads) < 2:
            n, clusters = 1, [0]
        else:    
            n, clusters = self.get_img_clusters(heads, lang)
        nodes = range(n)
        fixedpos = {person: (0.5, 0.5)}
        for node in nodes:
            G.add_edge(person,node)
            fixedpos.update({node: (random.random(), random.random()) })
            [G.add_edge(node, word) for ind, word in enumerate(heads) if clusters[ind] == node]
        remove = [node for node,degree in G.degree().items() if type(node) == int and degree == 0]
        G.remove_nodes_from(remove)
        [fixedpos.pop(i) for i in remove]
        pos = networkx.spring_layout(G, fixed = fixedpos.keys(), pos = fixedpos)
        self.draw_graph_on_sep_window(G, networkx.draw_networkx, pos=pos)
        networkx.draw_networkx(G, pos=pos)
        
        plt.show()
        
    def draw_graph_on_sep_window(self, G, draw_func, **kwargs):
        
        t = tk.Toplevel(self)
        t.wm_title("Graph")
        #root.bind("<Destroy>", destroy)


        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)

        ######################
        # the networkx part
        draw_func(G, ax=a, **kwargs)
        ######################

        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(f, master=t)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg( canvas, t )
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def get_img_clusters(self, heads, lang, max_k = 5):
        X = self.w2v_models[lang][heads]
        vector_pairs = tuple(combinations(range(X.shape[0]), 2))
        best_k, best_res, min_f = None, None, inf
        max_k = min(max_k, X.shape[0])
        for k in range(1, max_k + 1):
            clstr = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters = True)
            res = clstr.cluster(X, assign_clusters=True)
            f0 = [nltk.cluster.util.cosine_distance(X[x[0]], X[x[1]]) for x in vector_pairs
            if res[x[0]] == res[x[1]]]
            f1 = [nltk.cluster.util.cosine_distance(X[x[0]], X[x[1]]) for x in vector_pairs
            if res[x[0]] != res[x[1]]]
            f = sum(f0) * len(f1) / ( len(f0) * sum(f1) )
            if f < min_f:
                best_k, best_res, min_f = k, res, f
        return best_k, best_res
    
    def process_corp(self):
        for key, group in groupby(self.corpora[self.active_corp].values(), lambda text: langid.classify(text)[0]):
            if key == 'de':
                text_split, coref_model, ne_list, process = text_de_split, nnlp_de, ne_de, process_de
            elif key == 'en':
                text_split, coref_model, ne_list, process = text_en_split, nnlp, ne_en, process_en
            else:
                text_split, coref_model, ne_list, process = text_ru_split, parse_rus_texts, ne_ru, process_rus
            for text in group:
                process(text, text_split, ne_list)
            training_sents = [ [w[2].lower() for w in sent] for txt in text_split for sent in txt if sent]
            lemmas = {w[0]: w[2].lower() for txt in text_split for sent in txt for w in sent if sent}
            try:
                if key not in self.w2v_models:  self.w2v_models[key] = word2vec.Word2Vec.load('{}.bin'.format(key))
                len_before = len(self.w2v_models[key].wv.vocab)
                self.w2v_models[key].build_vocab(training_sents, update = True)
                self.w2v_models[key].train(training_sents, total_examples=self.w2v_models[key].corpus_count, epochs=self.w2v_models[key].iter)
                assert len_before < len(self.w2v_models[key].wv.vocab)
            except (ValueError, AssertionError, UnpicklingError):
                self.w2v_models[key] = word2vec.Word2Vec(training_sents, size=50, window=4, min_count=1 )
            del training_sents
            ne_clusts = clust_ne(ne_list)
            if not ne_clusts:
                continue
            clusts_to_ne = {v[0]:k for k,v in ne_clusts.items()}
            rel_and_im = []
            for ind_, text in enumerate(text_split):
                actors_clustered = coreference(self.w2v_models[key], coref_model, key, text, ne_clusts)
                if key == 'ru':
                    rels = list(get_relations_from_ru(text, actors_clustered, lemmas))
                else:
                    rels = list(get_relations_from(text, actors_clustered, key, lemmas))
                for rel_ind, rel in enumerate(rels):
                    sub = rel[0][0] if type(rel[0]) == tuple else rel[0]
                    ob = rel[1][0] if type(rel[1]) == tuple else rel[1]
                    rels[rel_ind].extend([sub, ob])
                    rels[rel_ind][0] = actors_clustered[rel[0]]
                    if len(rels[rel_ind][0])== 2 and type(rels[rel_ind][0]) == tuple:
                        rels[rel_ind][0], rels[rel_ind][1] = rels[rel_ind][0][0][0], rels[rel_ind][0][1][0]
                        continue
                    rels[rel_ind][0] = rels[rel_ind][0][0]
                    try:
                        rels[rel_ind][1] = actors_clustered[rel[1]]
                        if len(rels[rel_ind][1]) == 2 and type(rels[rel_ind][1]) == tuple:
                            rel_and_im.append([rels[rel_ind][0], rels[rel_ind][1][1][0], rels[rel_ind][2], rels[rel_ind][3], rels[rel_ind][4], rels[rel_ind][5]])
                            rels[rel_ind][1] = rels[rel_ind][1][0][0]
                            continue
                        rels[rel_ind][1] = rels[rel_ind][1][0]
                    except KeyError:
                        rels[rel_ind][1] = '-'
                    
                    
                rel_and_im.extend(rels)
            yield (key, (rel_and_im, ne_clusts, clusts_to_ne))
    
    def process_texts(self):
        
        self.tab_table = ttk.Frame(self.tabs, style='Menu.TFrame')
        self.tabs.add(self.tab_table, text='Table')
        
        self.kn_window = ttk.Frame(self.tab_table, width = self.winfo_screenwidth() - 200, height = self.winfo_screenheight(),
                                     style = 'CL.TFrame')
        self.kn_window.pack()
        self.kn_window.pack_propagate(0)

        self.kn = ttk.Treeview(self.kn_window)
        self.kn['columns'] = ("object", "relation")
        self.kn.column("object", width=100 )
        self.kn.column("relation", width=220 )
        self.kn.heading("object", text="object")
        self.kn.heading("relation", text="relation")
        self.proc_res = {rel[0]: rel[1] for rel in self.process_corp()}
        for key, data in self.proc_res.items():
            texts, ne_clusts, cl_to_name = data
            for text_item in texts:
                sub = '{} ({})'.format(cl_to_name[text_item[0]], text_item[-2]) if text_item[-2] not in ne_clusts else cl_to_name[text_item[0]]
                if text_item[1] == '-':
                    obj = text_item[1]
                elif text_item[-1] in ne_clusts:
                    obj = cl_to_name[text_item[1]]
                else:
                    obj = '{} ({})'.format(cl_to_name[text_item[1]], text_item[-1])
                self.kn.insert("", 0,  text = sub, values=(obj, text_item[2]))
                
        self.scbVDirSel = ttk.Scrollbar(self.kn,
                           orient=VERTICAL,
                           command=self.kn.yview)
        self.kn.configure(yscrollcommand=self.scbVDirSel.set)
        self.scbVDirSel.pack(side="right", fill="y")
        self.kn.pack(fill = BOTH, expand = True)
                
                
        self.analyze.add_command(label='Get personal graph of..', command = lambda: self.graph_window(1))
        self.analyze.add_command(label='Get graph of relations', command = lambda: self.graph_window(0))
        self.analyze.add_command(label='Get personal relations of...', command = lambda: self.graph_window(1,1))
        
    def graph_window(self, personal, with_relations = False):
        find_heads = lambda lang, cluster: [i[3].lower() for i in self.proc_res[lang][0] if i[0] == cluster]
        def change_personal_values(personal, lang, with_rels, checkbuttons):
            if not personal: return
            self.cb.grid_forget()
            self.cb = ttk.Combobox(t, values=tuple(self.proc_res[lang.get()][2].values()))
            self.cb.grid(row = 1, column = 2)
            if with_rels:
                [ch.grid_forget() for ch in checkbuttons]
                while checkbuttons: checkbuttons.pop()
                for key,val in self.proc_res[lang.get()][2].items():
                    checkbuttons.append(ttk.Checkbutton(t, text=val, onvalue=key+1, offvalue=None))
                    checkbuttons[-1].grid(row = 3 + (key//3), column = key % 3, pady = 3)
                self.window_button.grid_forget()
                self.window_button.grid(row = 4 + (len(checkbuttons)//3), column = 2)
            t.update()
            
        def draw_graph(lang, person, with_relations, checkbuttons):
            if person:
                cluster = self.proc_res[lang][1][person][0]
                heads = find_heads(lang, cluster)
                if not heads:
                    msg.showinfo('Error', 'Nothing to show conserning this person')
                    return
                if with_relations:
                    others = [i.cget('onvalue')-1 for i in checkbuttons if 'selected' in i.state() and i.cget('onvalue')-1 != cluster]
                    self.get_precise_rels(cluster, others, lang)
                self.get_personal_graph(heads, lang, person)
            else:
                self.get_relations_graph(lang)
            
        lang = tk.StringVar()
        t = tk.Toplevel(self)
        self.cb = tk.StringVar()
        title = 'Personal ' if personal else 'Relation '
        t.wm_title(title + "graph parameters")
        checkbuttons = []
        if 'ru' in self.w2v_models:
            self.rbutton_ru=tk.Radiobutton(t,text='Russian',variable=lang,value='ru', command = lambda: change_personal_values(personal, lang, with_relations, checkbuttons))
            self.rbutton_ru.grid(row = 2, column = 1, pady = 3)
        if 'en' in self.w2v_models:
            self.rbutton_en=tk.Radiobutton(t,text='English',variable=lang,value='en', command = lambda: change_personal_values(personal, lang, with_relations, checkbuttons))
            self.rbutton_en.grid(row = 2, column = 2, pady = 3)
        if 'de' in self.w2v_models:
            self.rbutton_de=tk.Radiobutton(t,text='German',variable=lang,value='de', command = lambda: change_personal_values(personal, lang, with_relations, checkbuttons))
            self.rbutton_de.grid(row = 2, column = 3, pady = 3)
        if personal:
            values = () if not lang.get() else tuple(self.proc_res[lang.get()][2].values())
            self.cb = ttk.Combobox(t, values=values)
            self.cb.grid(row = 1, column = 2)
            
        
        self.window_button = tk.Button(t, text="Get a graph", 
                                command=lambda: draw_graph(lang.get(), self.cb.get(), with_relations, checkbuttons))
        self.window_button.grid(row = 4 + (len(checkbuttons)//3), column = 2)
        
        
        
    def update_filelist(self, name, files, single = True):
        if not single:
            self.corpora[name] = files
            if name not in self.name_to_button:
                self.name_to_button[name] = Button(self.corp_window, text = name, command = lambda n = name: self.make_active(n), background = 'white', relief = 'flat')
                self.name_to_button[name].pack(anchor = W, padx = 15, pady = (15, 0))
            for file in files:
                if (name, file) not in self.name_to_button:
                    self.name_to_button[(name, file)] = Button(self.corp_window, text = file, command = lambda n = (name, file): self.make_active( n ), background = 'white', relief = 'flat')
                    self.name_to_button[(name, file)].pack()
        else:
            self.corpora[self.active_corp][name] = files
            if (self.active_corp, name) not in self.name_to_button:
                self.name_to_button[(self.active_corp, name)] = Button(self.corp_window, text = name, command = lambda n = (self.active_corp, name): self.make_active( n ), background = 'white', relief = 'flat')
            self.refresh_corplist()

    def refresh_corplist(self):

        for item in self.name_to_button:
            self.name_to_button[item].pack_forget()
        
        for corpus, files in self.corpora.items():
            self.name_to_button[corpus].pack(anchor = W, padx = 15, pady = (15, 0))
            for name in files:
                self.name_to_button[(corpus, name)].pack()
            

    def make_active(self, name):
        if name in self.corpora: # if a corpus was clicked
            self.active_corp = name
        else:
            self.active_corp = name[0]
            self.view_file(self.corpora[name[0]][name[1]])
            

    def view_file(self, text):
        self.text_field.delete('1.0', END)
        self.text_field.insert('1.0', text)
        
root = Application()
root.mainloop()


# In[ ]:



