import os
import sys

import pandas as pd

conditions = {

    'that_no-gap_subj' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'that_gap_subj' : ['Prefix', 'that', 'apositive', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_no-gap_subj' : ['Prefix', 'wh-subj', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_gap_subj' : ['Prefix', 'wh-subj', 'apositive', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],

    'that_no-gap_obj' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'that_gap_obj' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'Prep', 'NP3', 'End'],
    'what_no-gap_obj' : ['Prefix', 'wh-obj', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_gap_obj' : ['Prefix', 'wh-obj', 'apositive', 'NP1', 'Verb', 'Prep', 'NP3', 'End'],

    'that_no-gap_pp' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'that_gap_pp' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'End'],
    'what_no-gap_pp' : ['Prefix', 'wh-prep', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_gap_pp' : ['Prefix', 'wh-prep', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'End'],
}

extra = {

    'that_no-gap_verb' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'that_gap_verb' : ['Prefix', 'that', 'apositive', 'NP1', 'NP2', 'Prep','NP3', 'End'],
    'what_no-gap_verb' : ['Prefix', 'wh-subj', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_gap_verb' : ['Prefix', 'wh-subj', 'apositive', 'NP1', 'NP2', 'Prep','NP3', 'End'],
  
    'that_no-gap_prep' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'that_gap_prep' : ['Prefix', 'that', 'apositive', 'NP1', 'Verb', 'NP2', 'NP3', 'End'],
    'what_no-gap_prep' : ['Prefix', 'wh-subj', 'apositive', 'NP1', 'Verb', 'NP2', 'Prep', 'NP3', 'End'],
    'what_gap_prep' : ['Prefix', 'wh-subj', 'apositive', 'NP1','Verb', 'NP2', 'NP3', 'End']
}

end_condition_included = False
autocaps = True

def expand_items(df):
    output_df = pd.DataFrame(rows(df))
    output_df.columns = ['sent_index', 'word_index', 'word', 'region', 'condition']
    return output_df

def rows(df):
    for condition in conditions:
        for sent_index, row in df.iterrows():
            word_index = 0
            for region in conditions[condition]:
                for word in row[region].split():
                    if autocaps and word_index == 0:
                        word = word.title()
                    yield sent_index, word_index, word, region, condition
                    word_index += 1
            if not end_condition_included:
                yield sent_index, word_index + 1, ".", "End", condition
                yield sent_index, word_index + 2, "<eos>", "End", condition
            
def main(filename):
    input_df = pd.read_excel(filename)
    output_df = expand_items(input_df)
    try:
        os.mkdir("tests")
    except FileExistsError:
        pass
    output_df.to_csv("tests/items.tsv", sep="\t")

if __name__ == "__main__":
    main(*sys.argv[1:])
