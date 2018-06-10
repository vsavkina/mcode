
# coding: utf-8

# In[ ]:

#de_ner training data
import pandas as pd


# In[ ]:

tr_csv = pd.read_table(r'C:\Users\user\mcode\GermEval2014_complete_data\GermEval2014_complete_data\NER-de-train.tsv', encoding = 'utf8',
                      quoting=3, error_bad_lines=False)


# In[ ]:

#tr_csv.iloc[:, [2,1]]
#tr_csv
tr_csv.iloc[:, [0]] == '#'


# In[ ]:

ix = tr_csv.eq('#').any(axis=1)
sentences = []
begin_sent = [-1] + tr_csv.iloc[:,[0]][ix].index.tolist()
for index in range(1, len(begin_sent)):
    sentences.append(tr_csv.iloc[begin_sent[index-1]+1:begin_sent[index],:])
sentences.append(tr_csv.iloc[begin_sent[index]+1:,:])


# In[ ]:

"""def mjoin(mtable):
    count = 0
    cur = ''
    entities = []
    res = ''
    flag = False
    right_quote_expected = False
    for i in range(len(mtable)):
        w = mtable.iat[i,1]
        marker = mtable.iat[i,2]
        #finding persons
            
        if 'PER' in marker:
            
            flag = True
            
            cur += w if not cur else ' ' + w
            
        elif flag:
            tup = (count, count + len(cur), 'PERSON')
            entities.append(tup)
            flag = False
            cur = ''
    
        #building a sentence
        if w == '"':
            if right_quote_expected:
                res+=w 
            else:
                res += ' '+w
                if flag and marker == 'B-PER':
                    count+=1
            right_quote_expected = not right_quote_expected
        elif res and res[-1] == '"':
            if right_quote_expected or w in string.punctuation:
                res+=w 
            else:
                res+=' '+w
                if flag and marker == 'B-PER':
                    count += 1
        elif w in string.punctuation or not res or (len(res) > 1 and res[-2].isalpha() and res[-1] == '-'):
            res += w
        else:
            res += ' '+w
            if flag and marker == 'B-PER':
                count += 1
            
        if not flag:
            count = len(res)
    return (res, {'entities' : entities} )
    """
TRAIN_DATA = []
for i, sent in enumerate(sentences):
    if i % 1000 == 0:
        print(i)
    try:
        TRAIN_DATA.append(mjoin(sentences[i]))
    except TypeError as e:
        print('Error:', e, ' ', i)


# In[ ]:

td = [i for i in TRAIN_DATA if i[1]['entities']] + [i for i in TRAIN_DATA if not i[1]['entities']][:1000]


# In[ ]:

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy

import time

def main(model=None, output_dir=r"C:\Users\user\AppData\Local\conda\conda\envs\my_root\lib\site-packages\xx", n_iter=30):
    """Load the model, set up the pipeline and train the entity recognizer."""
    timeout = time.time() + 60*60*4 / 30
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('xx')  # create blank Language class
        print("Created blank 'xx' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')
    print("built-in pipeline components added")

    # add labels
    for _, annotations in td:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            print(itn, timeout - time.time())
            random.shuffle(td)
            losses = {}
            for text, annotations in td:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
                if time.time() > timeout:
                    timeout = time.time() + 60*60*4 / 30
                    break
            print(losses)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    random.shuffle(td)
    # test the trained model
    for text, _ in td[:20]:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in td:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


#main()

output_dir = Path(r"C:\Users\user\AppData\Local\conda\conda\envs\my_root\lib\site-packages\xx")
if not output_dir.exists():
    output_dir.mkdir()
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

