import os
import sys

import pandas as pd

conditions = {
    'that_no-gap_obj_no-mod': ['prefix', 'that', 'subj', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_obj_no-mod' : ['prefix', 'that', 'subj', 'verb', 'to', 'goal', 'temporal_modifier'],
    'wh_no-gap_obj_no-mod': ['prefix', 'obj wh', 'subj', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_obj_no-mod': ['prefix', 'obj wh', 'subj', 'verb', 'to', 'goal', 'temporal_modifier'],
    'that_no-gap_goal_no-mod': ['prefix', 'that', 'subj', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_goal_no-mod' : ['prefix', 'that', 'subj', 'verb', 'object', 'to', 'temporal_modifier'],
    'wh_no-gap_goal_no-mod': ['prefix', 'goal wh', 'subj', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_goal_no-mod': ['prefix', 'goal wh', 'subj', 'verb', 'object', 'to', 'temporal_modifier'],

    'that_no-gap_obj_short-mod': ['prefix', 'that', 'subj', 'short modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_obj_short-mod' : ['prefix', 'that', 'subj', 'short modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'wh_no-gap_obj_short-mod': ['prefix', 'obj wh', 'subj', 'short modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_obj_short-mod': ['prefix', 'obj wh', 'subj', 'short modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'that_no-gap_goal_short-mod': ['prefix', 'that', 'subj', 'short modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_goal_short-mod' : ['prefix', 'that', 'subj', 'short modifier', 'verb', 'object', 'to', 'temporal_modifier'],
    'wh_no-gap_goal_short-mod': ['prefix', 'goal wh', 'subj', 'short modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_goal_short-mod': ['prefix', 'goal wh', 'subj', 'short modifier', 'verb', 'object', 'to', 'temporal_modifier'],

    'that_no-gap_obj_med-mod': ['prefix', 'that', 'subj', 'medium modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_obj_med-mod' : ['prefix', 'that', 'subj', 'medium modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'wh_no-gap_obj_med-mod': ['prefix', 'obj wh', 'subj', 'medium modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_obj_med-mod': ['prefix', 'obj wh', 'subj', 'medium modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'that_no-gap_goal_med-mod': ['prefix', 'that', 'subj', 'medium modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_goal_med-mod' : ['prefix', 'that', 'subj', 'medium modifier', 'verb', 'object', 'to', 'temporal_modifier'],
    'wh_no-gap_goal_med-mod': ['prefix', 'goal wh', 'subj', 'medium modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_goal_med-mod': ['prefix', 'goal wh', 'subj', 'medium modifier', 'verb', 'object', 'to', 'temporal_modifier'],

    'that_no-gap_obj_long-mod': ['prefix', 'that', 'subj', 'long modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_obj_long-mod' : ['prefix', 'that', 'subj', 'long modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'wh_no-gap_obj_long-mod': ['prefix', 'obj wh', 'subj', 'long modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_obj_long-mod': ['prefix', 'obj wh', 'subj', 'long modifier', 'verb', 'to', 'goal', 'temporal_modifier'],
    'that_no-gap_goal_long-mod': ['prefix', 'that', 'subj', 'long modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'that_gap_goal_long-mod' : ['prefix', 'that', 'subj', 'long modifier', 'verb', 'object', 'to', 'temporal_modifier'],
    'wh_no-gap_goal_long-mod': ['prefix', 'goal wh', 'subj', 'long modifier', 'verb', 'object', 'to', 'goal', 'temporal_modifier'],
    'wh_gap_goal_long-mod': ['prefix', 'goal wh', 'subj', 'long modifier', 'verb', 'object', 'to', 'temporal_modifier']
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
