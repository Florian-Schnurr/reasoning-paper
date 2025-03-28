
import pandas as pd
import numpy as np
import spacy
from spacy.language import Language
from spacy.symbols import ORTH
spacy.cli.download('en_core_web_sm')
import json
from more_itertools import pairwise, windowed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch
from torch import nn
import warnings
import os
import logging
import time
from bertopic import BERTopic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["OMP_NUM_THREADS"] = "1"
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    

def init():
    """
    This function is called when the container is initialized/started, typically after creation/update of the deployment.
    The logic here caches the models and auxiliary files in memory.
    """
    global segmenters
    global nlp
    nlp = spacy.load('en_core_web_sm')
    segmenters_df = pd.read_csv("../models/segmentation/discourse_segmenters.csv", encoding="utf-8")
    segmenters = segmenters_df.phrase[segmenters_df['Include'] != 'N'].to_list()

    global subj1_tokenizer
    subj1_tokenizer = AutoTokenizer.from_pretrained("../models/mdebertav3-subjectivity-english/")
    global subj1_model
    subj1_model = AutoModelForSequenceClassification.from_pretrained("../models/mdebertav3-subjectivity-english")

    global subj2_tokenizer
    subj2_tokenizer = AutoTokenizer.from_pretrained("../models/BiBert-Subjectivity")
    global subj2_model
    subj2_model = AutoModelForSequenceClassification.from_pretrained("../models/BiBert-Subjectivity")

    global cert_tokenizer
    cert_tokenizer = AutoTokenizer.from_pretrained("../models/sentence-level-certainty")
    global cert_model
    cert_model = AutoModelForSequenceClassification.from_pretrained("../models/sentence-level-certainty")

    global sentiment_model
    sentiment_model_path = "../models/twitter-xlm-roberta-base-sentiment"
    sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)
 
    global extra_model
    extra_model = BertClassifier()
    extra_model.load_state_dict(torch.load("../models/reasoning-extra/extra_reasoning_model_2.pth", weights_only=True, map_location=device))
    extra_model.eval()
    global extra_tokenizer
    extra_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    global nli1_model
    nli1_path = "../models/argument-relation-mining-onnx"
    global nli1_tokenizer
    nli1_tokenizer = AutoTokenizer.from_pretrained(nli1_path)
    nli1_model = ORTModelForSequenceClassification.from_pretrained(nli1_path)

    global nli2_model
    nli2_model = CrossEncoder("../models/nli-MiniLM2-L6-H768")

    global d
    d = {}
    global dn
    dn = {}

    logging.info("Init complete")


def r_segmentation(text):
    """
    This function splits the text into argumentation units, and writes them into a simple DataFrame.
    Splitting is done via a list of common idea boundary phrases.
    """
    current_case = []
    for x in segmenters:
        current_case = [{ORTH: x}]
        nlp.tokenizer.add_special_case(x, current_case)

    pattern_docs = []
    for segmenter in segmenters:
        pattern_docs.append(segmenter)
        pattern_docs.append(segmenter[:1].upper() + segmenter[1:])

    pattern_docs = list(set(pattern_docs))

    if "set_sent_starts" not in nlp.pipe_names:
        @Language.component("set_sent_starts")
        
        def set_sent_starts(doc):
            for token in doc:
                if token.text in pattern_docs:
                    doc[token.i].is_sent_start = True
            return doc

        nlp.add_pipe("set_sent_starts", before='parser')

    doc = nlp(text)
    arg_units = []
    temp_string = ''
    idea_cutoff = 3

    for sent in doc.sents:
        if len(sent) < idea_cutoff:
            temp_string = ''.join([temp_string, str(sent), ' '])
            continue
        else:
            if temp_string != '':
                next_sentence = ''.join([temp_string, str(sent)])
                temp_string = ''
            else:
                next_sentence = str(sent)  
                
            arg_units.append(next_sentence)

    df = pd.DataFrame(arg_units, columns=['segment'])
    return df


def lexical_indicators(text):
    """
    This function looks for the same patterns used for segmentation, as an added indicator of reasoning-style language being used.
    """
    if any(str(i).lower() in text for i in segmenters):
        return True
    else:
        return False
    

def subjectivity1(text):
    """
    The primary subjectivity model comes from: GroNLP/mdebertav3-subjectivity-english on Hugging Face
    """
    inputs = subj1_tokenizer(text, return_tensors="pt")
    #inputs = inputs.to(device)
    with torch.no_grad():
        logits = subj1_model(**inputs).logits
    return float(torch.nn.functional.softmax(logits, dim=1)[0][1])


def subjectivity2(text):
    """
    The secondary subjectivity model comes from: HCKLab/BiBert-Subjectivity
    """
    inputs = subj2_tokenizer(text, return_tensors="pt")
    #inputs = inputs.to(device)
    with torch.no_grad():
        logits = subj2_model(**inputs).logits
    return float(torch.nn.functional.softmax(logits, dim=1)[0][1])


def certainty(text):
    """
    A model looking for certainty qualifiers (e.g. "probably", "almost certainly").
    Intended use is to check if the level of certainty is appropriate for the stage of argumentation.
    """

    inputs = cert_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = cert_model(**inputs).logits
    
    return float(logits - 4.4)


def predict_extra_reasoning(text):
    model = extra_model
    text_dict = extra_tokenizer(
        text,
        padding = 'max_length',
        max_length = 256,
        truncation = True,
        return_tensors = 'pt'
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    
    with torch.no_grad():
        mask = text_dict['attention_mask'].to(device)
        input_id = text_dict['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        probabilities = probs[0].cpu().numpy()

        evidence_prob = probabilities[0]
        good_prob = probabilities[1]

    return good_prob


def predict_extra_evidence(text):
    model = extra_model
    text_dict = extra_tokenizer(
        text,
        padding = 'max_length',
        max_length = 256,
        truncation = True,
        return_tensors = 'pt'
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    
    with torch.no_grad():
        mask = text_dict['attention_mask'].to(device)
        input_id = text_dict['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        probabilities = probs[0].cpu().numpy()

        evidence_prob = probabilities[0]
        good_prob = probabilities[1]

    return evidence_prob


def nli1(a,b):
    inputs = nli1_tokenizer(a, b, return_tensors='pt')
    outputs = nli1_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=1).argmax()

    if predictions == 0:
        result = 'Inference'
    elif predictions == 1:
        result = 'Conflict'
    elif predictions == 2:
        result = 'Rephrase'
    elif predictions == 3:
        result = 'NoRelation'
    else:
        result = 'ERROR'

    return result


def nli2(a,b):
    scores = nli2_model.predict((a, b))
    label_mapping = ["Conflict", "Inference", "NoRelation"]
    label = label_mapping[scores.argmax()]
    return label


def r_nli(df):
    pairwise_nli = {}
    for a,b in pairwise(df['segment']):
        prediction1 = nli1(a,b)
        prediction2 = nli2(a,b)
        pairwise_nli[b] = {"pairwise_1": prediction1, "pairwise_2": prediction2}

    df_pairs = pd.DataFrame.from_dict(pairwise_nli, orient='index')
    df_pairs.reset_index(inplace=True)
    df_pairs.rename(columns={'index': 'segment'}, inplace=True)

    pairwise_reverse_nli = {}
    for a,b in pairwise(df['segment']):
        prediction1 = nli1(b,a)
        prediction2 = nli2(b,a)
        pairwise_reverse_nli[b] = {"pairwise_reverse_1": prediction1, "pairwise_reverse_2": prediction2}

    df_reverse_pairs = pd.DataFrame.from_dict(pairwise_reverse_nli, orient='index')
    df_reverse_pairs.reset_index(inplace=True)
    df_reverse_pairs.rename(columns={'index': 'segment'}, inplace=True)

    tripwise_nli = {}
    for a,b,c in windowed(df['segment'], 3):
        prediction1 = nli1(a, c)
        prediction2 = nli2(a, c)
        tripwise_nli[c] = {"tripwise_1": prediction1, "tripwise_2": prediction2}

    df_trips = pd.DataFrame.from_dict(tripwise_nli, orient='index')
    df_trips.reset_index(inplace=True)
    df_trips.rename(columns={'index': 'segment'}, inplace=True)

    df_combined = df.merge(df_pairs,on='segment',how='left').merge(df_reverse_pairs,on='segment', how='left').merge(df_trips,on='segment', how='left')

    return df_combined


def combine_subj(row):
    if row['subj1'] > 0.3 and row['subj2'] > 0.53:
        subj = 1
    elif row['subj1'] < 0.32 and row['subj2'] < 0.56:
        subj = -1
    else:
        subj = (row['subj1'] - 0.3) + (row['subj2'] - 0.53)

    return subj


def sent_mod(df):
    df['sent_mod'] = 0
    for i in range(2, len(df)):
        if (df.loc[i-1, 'sent_score'] > 0.8) & (df.loc[i-1, 'sent_label'] in ['positive', 'negative']):
            if (df.loc[i, 'pairwise_1'] not in ['Conflict', 'Inference']):
                df.loc[i, 'sent_mod'] = df.loc[i, 'sent_mod'] - 1


        if (df.loc[i, 'sent_score'] > 0.8) & (df.loc[i, 'sent_label'] in ['positive', 'negative']):
            if (df.loc[i, 'pairwise_reverse_1'] not in ['Conflict', 'Inference']):
                df.loc[i, 'sent_mod'] = df.loc[i, 'sent_mod'] - 1
    return df


def subj_mod(df):
    df['subj_mod'] = 0
    for i in range(2, len(df)):
        if (df.loc[i, 'pairwise_1'] == "Inference"):
            if df.loc[i-1, 'subj'] < -0.5:
                if df.loc[i, 'subj'] > 0.5:
                    df.loc[i, 'subj_mod'] -= 1
        if (df.loc[i, 'tripwise_1'] == "Inference"):
            if df.loc[i-2, 'subj'] < -0.5:
                if df.loc[i, 'subj'] > 0.5:
                    df.loc[i, 'subj_mod'] -= 1
    return df


def cert_mod(df):
    df['cert_mod'] = 0
    for i in range(2, (len(df))):
        if (df.loc[i-2, 'cert'] > 1.0):
            if (df.loc[i, 'tripwise_1']) == "Conflict":
                df.loc[i, 'cert_mod'] -= 1
        if (df.loc[i-2, 'cert'] < 0.0):
            if (df.loc[i, 'tripwise_1']) == "Conflict":
                df.loc[i, 'cert_mod'] += 1
        if (df.loc[i-1, 'cert'] > 1.0):
            if (df.loc[i, 'pairwise_1']) == "Conflict":
                df.loc[i, 'cert_mod'] -= 1
        if (df.loc[i-1, 'cert'] < 0.0):
            if (df.loc[i, 'pairwise_1']) == "Conflict":
                df.loc[i, 'cert_mod'] += 1
    return df


def r_mod(row):
    mod_a = 0.0
    mod_b = 0.0
    mod_c = 0.0
    mod_d = 0.0
    mod_e = 0.0

    if row['good_reasoning'] > 0.33:       
        mod_e += 0.05
    if row['evidence_given'] > 0.53:   
        mod_e += 0.05 

    length = len(row['segment'].split())
    length_mod = np.clip(length, 3, 10) / 10

    acceptable = ["Inference", "Conflict"]

    if row['pairwise_1'] == "Inference":
        if row['subj'] == -1: mod_a = 0.05
        elif row['subj'] == 1: mod_a = 0.03
        elif row['subj'] < -0.25: mod_a = 0.045
        elif row['subj'] > 0.25: mod_a = 0.035
        else: mod_a = 0.04

        mod_a += 0.02

        if row['pairwise_2'] == "Inference": mod_a += 0.01

    if row['pairwise_1'] == "Conflict":
        if row['subj'] == -1: mod_a = 0.05
        elif row['subj'] == 1: mod_a = 0.03
        elif row['subj'] < -0.25: mod_a = 0.045
        elif row['subj'] > 0.25: mod_a = 0.035
        else: mod_a = 0.04

        mod_a += 0.02

        if row['pairwise_2'] == "Conflict": mod_a += 0.01    

    else:
        if row['pairwise_1'] == "Rephrase":
            mod_a = -0.04
        else: mod_a = -0.035   

        if row['pairwise_2'] in acceptable:
            mod_a += 0.01

        if row['pairwise_reverse_1'] == "Inference": 
            mod_b = 0.05

            if row['pairwise_reverse_2'] == "Inference": mod_b += 0.01
        
        elif row['pairwise_reverse_2'] == "Inference":
            mod_b = 0.01

    if row['tripwise_1'] == "Inference":
        if row['subj'] == -1: mod_c = 0.05
        elif row['subj'] == 1: mod_c = 0.03
        elif row['subj'] < -0.25: mod_c = 0.045
        elif row['subj'] > 0.25: mod_c = 0.035
        else: mod_c = 0.04

        if row['tripwise_2'] == "Inference": mod_c += 0.01
        mod_c += 0.02

    elif row['tripwise_1'] == "Conflict":
        if row['subj'] == -1: mod_c = 0.05
        elif row['subj'] == 1: mod_c = 0.03
        elif row['subj'] < -0.25: mod_c = 0.045
        elif row['subj'] > 0.25: mod_c = 0.035
        else: mod_c = 0.04

        if row['tripwise_2'] == "NoRelation": mod_c += 0.01
        mod_c += 0.02

    else:
        if row['tripwise_1'] == "Rephrase":
            mod_c = -0.04
        else: mod_c = -0.035
    
    if row['subj'] == -1: mod_d = 0.005
    elif row['subj'] == 1: mod_d = -0.005
    elif row['subj'] < -0.25: mod_d = 0.003
    elif row['subj'] > 0.25: mod_d = -0.003
    else: mod_d = 0.0

    sent_mod = row['sent_mod'] * 0.005

    lex_mod = row['lexical_indicators'] * 0.005

    cert_mod = row['cert_mod'] * 0.005
    subj_mod = row['subj_mod'] * 0.005

    mod = (1.1 * mod_a + 0.8*(mod_b) + 0.5*(mod_c) + mod_d + 0.8 * mod_e + sent_mod + lex_mod + cert_mod + subj_mod) * length_mod
    
    return mod


def aggregate_stats(df):
    df['any_conflict'] = df[['other_comp', 'pairwise_1', 'pairwise_reverse_1', 'tripwise_1']].apply(lambda x: x.str.contains("Conflict", regex=False).any(), axis=1)
    df['any_inference'] = df[['other_comp', 'pairwise_1', 'pairwise_reverse_1', 'tripwise_1']].apply(lambda x: x.str.contains("Inference", regex=False).any(), axis=1)
    df['either'] = df['any_conflict'] | df['any_inference']

    num_topics = df.topic.nunique()
    num_inference = df.any_inference.sum()
    num_conflict = df.any_conflict.sum()
    num_either = df.either.sum()

    prop_either = num_either / len(df)
    balance = min(num_conflict, num_inference) / max(num_conflict, num_inference)

    df.loc[df.index[-1], 'num_topics'] = num_topics
    df.loc[df.index[-1], 'prop_either'] = prop_either
    df.loc[df.index[-1], 'balance'] = balance

    if num_topics > 1: 
        vc = df.groupby(by="topic").agg(
            inf = ("any_inference", "sum"),
            conf = ("any_conflict", "sum"),
            either = ("either", "sum"),
            num = ("segment", "count")
        ).reset_index()

        vc = vc.drop(vc[vc['topic'] == -1].index)
        vc.reset_index()

        largest_topic_share = vc['num'].max() / len(df)                  
        df.loc[df.index[-1], 'largest_topic_share'] = largest_topic_share

        topic_size_balance = vc.num.std()

        vc['topic_inference'] = vc['either'] / vc['num']
        max_topic_inference = vc['topic_inference'].max()
        df.loc[df.index[-1], 'max_topic_inference'] = max_topic_inference

        vc['topic_balance'] = vc[['inf', 'conf']].min(axis=1) / vc[['inf', 'conf']].max(axis=1)
        vc.replace([np.inf, -np.inf], 0, inplace=True)

        vc['differences'] = np.abs(vc['topic_balance'] - 1.0)
        nearest_value = vc['differences'].min()
        df.loc[df.index[-1], 'max_topic_balance'] = nearest_value
    else:
        df.loc[df.index[-1], 'largest_topic_share'] = 0.0
        df.loc[df.index[-1], 'max_topic_inference'] = 0.0
        df.loc[df.index[-1], 'max_topic_balance'] = 0.0

    return df


def r_mod_agg(df):
    try:
        last_agg_row = (df[~df['num_topics'].isnull()]).take([-1]).index[0]
    except:
        return df

    moda = -(df.iloc[last_agg_row].num_topics)/100
    modb = df.iloc[last_agg_row].largest_topic_share / 10

    modc = df.iloc[last_agg_row].prop_either - 0.7
    modd = df.iloc[last_agg_row].balance - 0.5 
    mode = df.iloc[last_agg_row].max_topic_balance - 0.7
    modf = df.iloc[last_agg_row].max_topic_inference - 0.7

    rmod_agg = np.clip((moda + modb + 1.5 * modc + 0.5 * modd + 0.4 * mode + 0.4 * modf), -1, 1)

    df.loc[last_agg_row, "rmod_agg"] = rmod_agg

    if len(df[~df['num_topics'].isnull()]) > 1:
        penu_agg_row = (df[~df['num_topics'].isnull()]).take([-2]).index[0]
        df.loc[last_agg_row, "prev_rmod_agg"] = df.loc[penu_agg_row, "rmod_agg"]
    else:
        df.loc[last_agg_row, "prev_rmod_agg"] = 0

    return df


def r_calculation_new(df):
    for i in range(len(df.r_mod)-1):
        a = df.loc[i, 'r']
        b = df.loc[i+1, 'r_mod']
        if 0.5 - a >= 0:
            if b > 0:
                c = b
            else:
                c = (0.5 - abs(0.5 - a)) * 2 * b
        elif 0.5 - 1 < 0:
            if b < 0:
                c = b
            else:
                c = (0.5 - abs(0.5 - a)) * 2 * b

        df.loc[i+1, 'r'] = a + c
    
    return df


def r_calculation_ongoing(df):
    for i in range(len(df.r_mod)-1):
        a = df.loc[i, 'r']
        b = df.loc[i+1, 'r_mod']

        if not pd.isna(df.at[i+1, 'rmod_agg']):
            c = df.loc[i+1, "rmod_agg"]
            d = df.loc[i+1, "prev_rmod_agg"]

            agg_r = c - d 
            len_mod = np.clip((len(df) / 500), 0, 0.5)
            rev_len_mod = 1 - len_mod
            mod = np.clip(((rev_len_mod * b) + (len_mod * agg_r)), -1, 1)

            if 0.5 - a >= 0:
                if mod > 0:
                    x = mod
                else:
                    x = (0.5 - abs(0.5 - a)) * 2 * mod
            elif 0.5 - 1 < 0:
                if mod < 0:
                    x = mod
                else:
                    x = (0.5 - abs(0.5 - a)) * 2 * mod

            df.loc[i+1, 'r'] = a + x

        else:
            if 0.5 - a >= 0:
                if b > 0:
                    x = b
                else:
                    x = (0.5 - abs(0.5 - a)) * 2 * b
            elif 0.5 - 1 < 0:
                if b < 0:
                    x = b
                else:
                    x = (0.5 - abs(0.5 - a)) * 2 * b

            df.loc[i+1, 'r'] = a + x
    
    return df


def new_meeting(text):
    df = r_segmentation(text)
    df['lexical_indicators'] = df['segment'].apply(lexical_indicators)
    df['cert'] = df['segment'].apply(lambda x: certainty(x))
    df['subj1'] = df['segment'].apply(subjectivity1)
    df['subj2'] = df['segment'].apply(subjectivity2)
    df['subj'] = df.apply(combine_subj, axis=1)
    df['sent_label'] = df['segment'].apply(lambda x: sentiment_model(x)[0]['label'])
    df['sent_score'] = df['segment'].apply(lambda x: sentiment_model(x)[0]['score'])
    df['good_reasoning'] = df['segment'].apply(lambda x: predict_extra_reasoning(x))
    df['evidence_given'] = df['segment'].apply(lambda x: predict_extra_evidence(x))

    df['topic'] = -1
    df['topic_state'] = False
    df['other_comp'] = np.nan
    df['all_comp'] = np.nan
    
    df = r_nli(df)
    df = sent_mod(df)
    df = cert_mod(df)
    df = subj_mod(df)
    df['r_mod'] = df.apply(r_mod, axis=1)
    df.loc[0, 'r'] = 0.5

    return df


def existing_meeting(text, df):
    new_text = df.tail(-1).reset_index().loc[0, 'segment'] + ' ' + text
    new_segs = r_segmentation(new_text)

    if new_segs.head(1).loc[0, 'segment'] == df.tail(-1).reset_index().loc[0, 'segment']:
        new_segs = r_segmentation(text)
        df = pd.concat([df, new_segs], axis=0, ignore_index=True)
    else:
        df = pd.concat([df.head(-1), new_segs], axis=0, ignore_index=True)

    topic_retrain_flag = False
    if len(df) > 20:
        if df['topic_state'][-10:].sum() == 0:
            global topic_model
            min_topic_size = np.clip((len(df) // 20), 3, 100)
            try:   
                topic_model = BERTopic(nr_topics = "auto", min_topic_size=min_topic_size)
                docs = df.segment.to_list()
                topics, probs = topic_model.fit_transform(docs)
                tops = pd.DataFrame({"segment": docs, "topic": topic_model.topics_})
                df = df.drop(columns='topic').join(tops.drop(columns='segment'))
                df['topic_state'] = True
                topic_retrain_flag = True
            except:
                df['topic'] = -1
                df['topic_state'] = False
                df['all_comp'] = np.nan
                df['latest_comp'] = np.nan
    else:
        df['topic'] = -1
        df['topic_state'] = False
        df['all_comp'] = np.nan
        df['latest_comp'] = np.nan

    for row in df.itertuples():
        i = getattr(row, "Index")
        if not pd.notnull(getattr(row, 'lexical_indicators')):
            df.loc[i, 'lexical_indicators'] = lexical_indicators(df.loc[i, 'segment'])

            df.loc[i, 'subj1'] = subjectivity1(df.loc[i, 'segment'])
            df.loc[i, 'subj2'] = subjectivity2(df.loc[i, 'segment'])

            sentiment = sentiment_model(df.loc[i, 'segment'])
            df.loc[i, 'sent_label'] = sentiment[0]['label']
            df.loc[i, 'sent_score'] = sentiment[0]['score']

            df.loc[i, 'cert'] = certainty(df.loc[i, 'segment'])

            df.loc[i, 'good_reasoning'] = predict_extra_reasoning(df.loc[i, 'segment'])
            df.loc[i, 'evidence_given'] = predict_extra_evidence(df.loc[i, 'segment'])
        
            a = df.loc[i-2, 'segment']
            b = df.loc[i-1, 'segment']
            c = df.loc[i, 'segment']

            df.loc[i, "pairwise_1"] = nli1(a,b)
            df.loc[i, "pairwise_2"] = nli2(a,b)
            df.loc[i, "pairwise_reverse_1"] = nli1(b,a)
            df.loc[i, "pairwise_reverse_2"] = nli2(b,a)
            df.loc[i, "tripwise_1"] = nli1(a,c)
            df.loc[i, "tripwise_2"] = nli2(a,c)

            if topic_retrain_flag == False:
                if df["topic_state"].sum() > 0:
                    df.loc[i, "topic"] = topic_model.transform(df.loc[i, "segment"])[0][0]
                    if i > 2:
                        if df.loc[i, 'topic'] != df.loc[i-1, 'topic'] and df.loc[i, 'topic'] != df.loc[i-2, 'topic']:
                            if (df.loc[i, 'pairwise_1'] in ['Rephrase', 'NoRelation']) and (df.loc[i, 'tripwise_1'] in ['Rephrase', 'NoRelation']):
                                dfcut = df.head(i+1)
                                if len(dfcut[dfcut["topic"] == dfcut.loc[i, "topic"]]) > 1:
                                    j = int(dfcut[dfcut["topic"] == dfcut.loc[i, "topic"]].take([-2]).index[0])
                                    x = dfcut.loc[j, 'segment']
                                    comp = nli1(x, c)
                                    dix = {j: comp}
                                    df.loc[i, "all_comp"] = json.dumps(dix)
                                    df.loc[i, "latest_comp"] = json.dumps(dix)

        if topic_retrain_flag and i > 2:
            if df.loc[i, 'topic'] != df.loc[i-1, 'topic'] and df.loc[i, 'topic'] != df.loc[i-2, 'topic']:
                if (df.loc[i, 'pairwise_1'] in ['Rephrase', 'NoRelation']) and (df.loc[i, 'tripwise_1'] in ['Rephrase', 'NoRelation']):
                    dfcut = df.head(i+1)
                    if len(dfcut[dfcut["topic"] == dfcut.loc[i, "topic"]]) > 1:
                        j = int(dfcut[dfcut["topic"] == dfcut.loc[i, "topic"]].take([-2]).index[0])
                        if pd.notnull(getattr(row, 'all_comp')):
                            previous = json.loads(df.loc[i, 'all_comp'])
                            if j in previous:
                                df.loc[i, 'latest_comp'] = json.dumps({j: previous[j]})
                            if str(j) in previous:
                                df.loc[i, 'latest_comp'] = json.dumps({j: previous[str(j)]})
                                continue
                            else:
                                x = df.loc[j, 'segment']
                                c = df.loc[i, 'segment']
                                comp = nli1(x, c)
                                dix = {j: comp}
                                previous[j] = comp
                                df.loc[i, "all_comp"] = json.dumps(previous)
                                df.loc[i, "latest_comp"] = json.dumps(dix)
                        else:
                            x = df.loc[j, 'segment']
                            c = df.loc[i, 'segment']
                            comp = nli1(x, c)
                            dix = {j: comp}
                            df.loc[i, "all_comp"] = json.dumps(dix)
                            df.loc[i, "latest_comp"] = json.dumps(dix)

        if (df.loc[i, 'pairwise_1'] == "Inference") or (df.loc[i, 'pairwise_reverse_1'] == "Inference"):
            if df.loc[i, 'topic'] == df.loc[i-1, 'topic']:
                df.loc[i, 'coherence'] = 1
            else:
                df.loc[i, 'coherence'] = -1
        else:
            df.loc[i, 'coherence'] = 0

        if df.loc[i, 'tripwise_1'] == "Inference":
            if df.loc[i, 'topic'] == df.loc[i-2, 'topic']:
                df.loc[i, 'coherence'] += 1
            else:
                df.loc[i, 'coherence'] -= 1

    return df


def reasoning(text, mcid):
    if mcid in d:
        df = existing_meeting(text, d[mcid][0])
        df['subj'] = df.apply(combine_subj, axis=1)
        df = subj_mod(df)
        df = sent_mod(df)        
        df = cert_mod(df)
        df['r_mod'] = df.apply(r_mod, axis=1)
        # apply r agg? - Only if there already has been an aggregation step (len(df) > 20)
        if len(df) > 20:
            df = aggregate_stats(df)
            df = r_mod_agg(df)
            df = r_calculation_ongoing(df)
        else:
            df = r_calculation_new(df)
        r = df.loc[df.index[-1], 'r']
        d[mcid] = [df, time.time()]

    else:
        if mcid in dn:
            #old_text = str(dn[mcid][0])
            old_text = ' '.join([str(item) for item in dn[mcid]])
            text = old_text + ' ' + text

        segs = r_segmentation(text)
        if len(segs) >= 3: 
            # If it's much bigger than 3, we need to split into some going through New Meeting and some through Existing Meeting.            
            if len(segs) > 20:
                old_text = ' '.join(segs['segment'][:18].tolist())
                new_text = ' '.join(segs['segment'][18:].tolist())

                df = new_meeting(old_text)
                df = r_calculation_new(df)
                r = df.loc[df.index[-1], 'r']
                d[mcid] = [df, time.time()]
                
                df = existing_meeting(new_text, d[mcid][0])
                df['subj'] = df.apply(combine_subj, axis=1)
                df = subj_mod(df)
                df = sent_mod(df)        
                df = cert_mod(df)
                df['r_mod'] = df.apply(r_mod, axis=1)
                df = aggregate_stats(df)
                df = r_mod_agg(df)
                df = r_calculation_ongoing(df)
                r = df.loc[df.index[-1], 'r']
                d[mcid] = [df, time.time()]

            else:
                df = new_meeting(text)
                df = r_calculation_new(df)
                r = df.loc[df.index[-1], 'r']
                d[mcid] = [df, time.time()]
        else:
            dn[mcid] = [text]
            r = 0.5

    for key,value in d.copy().items():
        now = time.time()
        last_update = value[-1]
        if (last_update < (now-600)):
            del d[key]

            if key in dn.keys():
                del dn[key]

    return r


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    Currently only the most recent Reasoning score is returned.
    """
    logging.info("Reasoning model: request received")
    text = str(json.loads(raw_data)["text"])
    mcid = str(json.loads(raw_data)["mcid"])

    r = reasoning(text, mcid)
    response = {"latest_r": r}
    logging.info(f"latest_r:  {r}")

    return response
