import plotly.graph_objects as go
import pandas as pd
import random
from collections import defaultdict


def get_color_hue(frequencies, colors):
    """ returns a tricolor tuple based on the specific word ratio by label and the indicated colors per label """
    # get total frequencies of specific word
    total_freq = sum(frequencies.values())
    first = 0
    second = 0
    third = 0
    # update each color value based on the frequency ratio of reach label
    print()
    for key, value in frequencies.items():
        if key in colors.keys():
            rgb_tuple = colors.get(key)
            first += (value / total_freq) * rgb_tuple[0]
            second += (value / total_freq) * rgb_tuple[1]
            third += (value / total_freq) * rgb_tuple[2]
    new_color = (first, second, third)
    return new_color


def map_colors(df, all_labels, label_color_dict, tricolor):
    """ returns a list of rgb colors based on the corresponding node label """
    color_lst = []
    lst_title = df['title'].unique().tolist()
    lst_words = df['words'].unique().tolist()
    # iterating through all node labels
    for label in all_labels:
        # check if the label is a part of the title column (src node)
        if label in lst_title:
            # if the corresponding label of the node is in the color dictionary, append that specific color, otherwise set it to back
            id_label = df[df['title'] == label]['label'].values[0]
            if id_label in label_color_dict.keys():
                color_lst.append(label_color_dict[id_label])
            else:
                color_lst.append((0, 0, 0))
        # check if the label is a part of the word column (targ node)
        if label in lst_words:
            word_freq_dict = defaultdict(dict)
            # returns df only one specific targ node label name
            df_word = df[df['words'] == label].reset_index()
            # mapping frequency of the word to the label of the word in the word frequency dictionary
            for i in df_word.index:
                if df_word.iloc[i]['label'] not in word_freq_dict.keys():
                    word_freq_dict[df_word.iloc[i]['label']] = df_word.iloc[i]['frequency']
                else:
                    word_freq_dict[df_word.iloc[i]['label']] += df_word.iloc[i]['frequency']
            # returning the corresponding hue based on the given frequency ratios
            color = get_color_hue(word_freq_dict, label_color_dict)
            color_lst.append(color)
    # changing format of rgb color to be usable by sankey diagram function
    color_lst = list(map(str, color_lst))
    color_lst = [tricolor + s for s in color_lst]
    return color_lst



def _code_mapping(df, src, targ):
    """ Maps labels / strings in src and target and converts them to integers 0, 1, 2, 3, ... """

    # extract distinct labels
    labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # define integer codes
    codes = list(range(len(labels)))

    # pair labels with list --> creating tuples
    lc_map = dict(zip(labels, codes))

    # in df, substitute codes for labels
    df = df.replace({src: lc_map, targ: lc_map})

    return df, labels


def make_sankey(df, src, targ, label_color_dict=None, tricolor_colormap=None, color_function=None, vals=None, **kwargs):
    """ Generate the sankey diagram """

    original_df = df
    df, labels = _code_mapping(df, src, targ)

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 30)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 1)

    if tricolor_colormap is None:
        tricolor = 'rgb'

    # if label color dict is inputted, return list of node colors based on the specified colors
    if label_color_dict:
        colors = map_colors(original_df, labels, label_color_dict, tricolor)
    # if label color dict not inputted, node colors will be randomized
    else:
        colors = None

    link = {'source': df[src], 'target': df[targ], 'value': values}
    node = {'label': labels, 'color': colors,
            'pad': pad, 'thickness': thickness,
            'line': {'color': line_color, 'width': line_width}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    fig.show()

