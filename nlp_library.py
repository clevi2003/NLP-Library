from collections import defaultdict, Counter, OrderedDict
from matplotlib import pyplot as plt, rcParams
import string
from gensim.parsing.preprocessing import remove_stopwords
import re
import nltk
import numpy as np
import datetime
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import sankey as sk

VOWELS = ["a", "i", "e", "o", "u", "y", "A", "E", "I", "O", "U", "Y"]


def map_parts_speech(filename):
    """

    :param filename: string
        the file has values separated by tabs
    :return: parts_speech_map: dict
        maps abbreviations to their full name
    """
    mapping_data = []
    with open(filename, "r") as infile:
        lines = infile.readlines()

        # creates a 2d list with each sublist being an abbreviation and its full name
        for line in lines:
            line = line.strip().split("\t")
            map_data = [line[1], line[2]]
            mapping_data.append(map_data)
    # maps each sublist to a dictionary entry
    parts_speech_map = {elem[0]: elem[1] for elem in mapping_data}
    return parts_speech_map


def zero_default_dict():
    return 0


def make_dict():
    return defaultdict(dict)


class Text:

    def __init__(self):
        """ Constructor """
        self.data = defaultdict(make_dict)  # extracted data (state)

    @staticmethod
    def _color_label(color_map, group):
        """

        :param color_map: dictionary
            maps group to desired group color
        :param group: string
            group label
        :return: tuple
            color and its associated group label for a plot legend
        """
        if group in color_map.keys():
            color = color_map[group]
            label = group
        else:
            color = (0, 0, 0)
            label = 'Other'
        return color, label

    @staticmethod
    def _flesch_kincaid_test(text):
        """

        :param text: string
            all text in a file
        :return: score: float
            readability score. Corresponds to a grade level required to read the text
        """
        # .39 * (total words/ total sentences) + 11.8 * (total syllables / total words) - 15.59
        # words with more syllables are harder to read than words with fewer syllables
        score = 0.0
        if len(text) > 0:
            score = (0.39 * len(text.split()) / len(text.split('.'))) + 11.8 * (
                    sum(list(map(lambda x: 1 if x in VOWELS else 0, text))) /
                    len(text.split())) - 15.59
        return score

    @staticmethod
    def _word_color(word, full_word_freq, full_word_group, group_color_map, time):
        """

        :param word: string
            word to find the RGB color for
        :param full_word_freq: dictionary
            {time_period:{word: count}}
        :param full_word_group: dictionary
            {time_period:{word:{group: count}}}
        :param group_color_map: dictionary
            maps each group to its associated color
        :param time: string
            time period of interest
        :return: tuple
            RGB tuple for the input word
        """
        color_index = {"red": 0, "green": 1, "blue": 2}

        rgb_word = [0, 0, 0]
        for group in group_color_map.keys():
            """
            for time_period in full_word_group.keys():
                if time == time_period:
            """
            if group in full_word_group[time][word].keys():
                group_freq = full_word_group[time][word][group]
                group_color = ((group_freq / full_word_freq[time][word]) * 255)
                rgb_word[color_index[group_color_map[group]]] = round(group_color)

        return tuple(rgb_word)

    def _save_results(self, year, label, title, results):
        """

        :param label: string
            group for the text
        :param title: string
            title of the text
        :param results: dict
            holds all data for the given text
        :return: None
        """
        for key, value in results.items():
            self.data[label][key][title] = value

    # keys are data categories
    # values are that data category for each file
    # label = A
    # results = {'wordcount': wcA, 'numwords': 25}
    # results = {'wordcount': wcB, 'numwords': 30}
    # data = {'wordcount':{'A':wcA, 'B':wcB},  'numwords':{'A':25, 'B':30}}
    # word count, num words, readability score, part of speech

    @staticmethod
    def _part_speech(sentences):
        """

        :param sentences: string
            all text in a file
        :return: dict
            frequencies of parts of speech in the input text
        """
        speech_parts = []
        for sentence in sentences.split("."):
            sentence = remove_stopwords(sentence).strip()
            # tag words with their part of speech
            sentence_speech = nltk.pos_tag(sentence.split(" "))
            # make like of parts of speech
            sentence_speech = [elem[1] for elem in sentence_speech]
            speech_parts += sentence_speech
        return Counter(speech_parts)

    def _default_parser(self, filename, year):
        """

        :param year: int
            year of the text file
        :param filename: string
            name of the relevant text file
        :return: results: dict
            holds all data for the given text
        """
        with open(filename, "r", encoding="unicode_escape") as infile:
            lines = infile.readlines()
        # make string for all text in file
        lines_string = " ".join(lines)
        # calculate readability score
        readability = self._flesch_kincaid_test(lines_string)
        # create frequency dict for parts of speech
        parts_of_speech = self._part_speech(lines_string)
        words = []
        for line in lines:
            # remove filler words
            stripped_line = remove_stopwords(line.strip().lower())
            stripped_line = stripped_line.replace("'", "")
            # splits on anything that's not a letter
            temp_words = re.split('[^a-zA-Z]', stripped_line)
            for word in temp_words:
                if len(word) > 0:
                    words.append(word)
        # creates dict with all relevant data and data statistics
        results = {
            'word count': Counter(words),
            'num words': len(words),
            "readability difficulty": readability,
            "parts of speech": parts_of_speech,
            "year": year
        }

        return results

    def load_text(self, filename, year=None, label=None, title=None, parser=None):
        """Registers the text file with the NLP framework"""

        if year is None:
            year = datetime.now().year

        if parser is None:
            results = self._default_parser(filename, year)
        else:
            results = parser(filename, year)

        if title is None:
            title = filename

        self._save_results(year, label, title, results)

    def frequency_filter(self, threshold, category):
        for group in self.data.keys():
            category_dict = self.data[group][category]
            for title, text_dict in category_dict.items():
                text_dict = {key: value for key, value in text_dict.items() if value >= threshold}
                category_dict[title] = text_dict
            self.data[group][category] = category_dict

    def rename_keys(self, category, map_dict):
        """

        :param category: string
            name of data statistic
        :param map_dict: dict
            maps key names to what they should be renamed as
        :return: None
        """
        # iterates over different text groups
        for group in self.data.keys():
            category_dict = self.data[group][category]
            # iterates over texts within a group
            for title, text_dict in category_dict.items():
                temp_dict = {}
                for sub_key in text_dict.keys():
                    # if the current key is in map_dict, populate temp_dict with the associated key name
                    if sub_key in map_dict.keys():
                        temp_dict[map_dict[sub_key]] = text_dict[sub_key]
                category_dict[title] = temp_dict
            self.data[group][category] = category_dict
    """
    def combine_groups(self, category, groups):
        combined_category = defaultdict(zero_default_dict)
        for group in groups:
            for title, category_data in self.data[group][category]:
                for data_point, frequency in category_data.items():
                    combined_category[word] += frequency
        return combined_category
    """

    def time_word_cloud(self, len_time_periods, min_year, max_year, groups):
        """

        :param len_time_periods: integer
            number of years in a given time period
        :param min_year: integer
            the earliest year to include in a time period
        :param max_year: integer
            the most recent year to include in a time period
        :param groups: list
            different group labels to include in the word cloud
        :return: None
            plots word clouds
        """
        # makes a list of time period ranges that correspond to each word cloud subplot
        periods = range(min_year, max_year, len_time_periods)
        period_nested = [[periods[i], periods[i + 1] - 1] for i in range(len(periods) - 1)]
        period_nested.append([period_nested[-1][1] + 1, max_year])

        # creates dict with time periods as keys and combined group dictionaries as values
        # group dictionaries have words as keys and their counts for all groups as values
        full_word_freq = defaultdict(dict)
        for time_period in period_nested:
            time_period_name = str(time_period[0]) + "-" + str(time_period[1])
            word_freq = defaultdict(zero_default_dict)
            # iterate over groups to combine frequencies of each group if they're in the given time period
            for group in groups:
                for title, year in self.data[group]["year"].items():
                    # check time period
                    if year in range(time_period[0], time_period[1] + 1):
                        for word, frequency in self.data[group]["word count"][title].items():
                            word_freq[word] += frequency
            # adds time period compiled data to overall dict
            # full_ word_freq = {time_period:{word: count}} 
            full_word_freq[time_period_name] = word_freq

        # creates same dict as above except with frequency counts separated by group
        full_word_group = defaultdict(dict)
        for time_period in period_nested:
            name = str(time_period[0]) + "-" + str(time_period[1])
            word_group_freq = defaultdict(dict)
            for group in self.data.keys():
                word_count_dict = self.data[group]["word count"]
                for title, year in self.data[group]["year"].items():
                    if year in range(time_period[0], time_period[1] + 1):
                        for word in word_count_dict[title].keys():
                            count = word_count_dict[title][word]
                            if word not in word_group_freq:
                                word_group_freq[word] = defaultdict(zero_default_dict)
                            # populate dict with word count frequency separated by group label
                            word_group_freq[word][group] += count
                # full_word_group = {time_period:{word:{group: count}}}
                full_word_group[name] = word_group_freq

        # make figure for subplots
        fig = plt.figure()
        # counter to keep track of position (necessary for subplot) when iterating by dictionary key
        counter = 1
        for time_period in full_word_freq.keys():
            def color_func(word, font_size, position, orientation, font_path, random_state):
                return self._word_color(word, full_word_freq, full_word_group,
                                        {"Democrat": "blue", "Republican": "red"}, time_period)

            # check if word frequencies for a time period are empty. If they are, don't make a plot
            if full_word_freq[time_period] == {}:
                continue
            wc = WordCloud(background_color="white").generate_from_frequencies(
                full_word_freq[time_period]
            )
            # plotting subplots based on dimensions of two columns and a variable number of rows
            ax = fig.add_subplot(round(len(full_word_freq) / 2), 2, counter)
            ax.title.set_text(time_period)
            ax.imshow(wc.recolor(color_func=color_func))
            ax.axis('off')
            counter += 1
        fig.suptitle("Common Words by Time Period", fontsize=15)
        fig.tight_layout(h_pad=.01, w_pad=1)
        plt.show()

    def sankey_diagram(self, min_common_words, label_color_dict, min_year=None, max_year=None, tricolor_colormap=None):
        """

        :param min_common_words: int
            minimum amount of common words
        :param label_color_dict: dict
            dict of inputted labels and corresponding rgb colors
        :param min_year: int
            optional parameter of minimum year
        :param max_year: int
            optional parameter of maximum year
        :param tricolor_colormap: str
            3 character string referring to the tricolor needed to map colors, default is "rgb"
        :return: None
        """
        # create empty df with specific column titles
        all_data_df = pd.DataFrame(columns=['title', 'words', 'frequency', 'label', 'year'])
        for group in self.data.keys():
            wc_dict = self.data[group]["word count"]
            year_dict = self.data[group]["year"]
            for titles, text_dict in wc_dict.items():
                for word, freq in text_dict.items():
                    for title, year in year_dict.items():
                        if title == titles:
                            corr_year = year
                        # create df with info for single txt file
                        curr_row = pd.DataFrame([{"title": re.sub('[^a-zA-Z]+', '', titles),
                                                  'words': word,
                                                  'frequency': freq,
                                                  'label': group,
                                                  'year': corr_year}])
                    # concat single file df to entire df
                    all_data_df = pd.concat([all_data_df, curr_row], ignore_index=True)
        # set min / max year constraints
        if min_year:
            all_data_df = all_data_df[all_data_df.year >= min_year]
        if max_year:
            all_data_df = all_data_df[all_data_df.year <= max_year]
        # group frequencies of individual word and common titles
        title_grouped_df = all_data_df.groupby(['title', 'words'])['frequency'].sum().reset_index(
            name='title frequency')
        all_data_df = all_data_df.merge(title_grouped_df, on=['title', 'words'])
        # get total frequencies for each word
        grouped_df = all_data_df.groupby(['words'])['frequency'].sum().reset_index(name='total frequency')
        all_data_df = all_data_df.merge(grouped_df, on='words')
        # filter on minimum number of common words
        all_data_df = all_data_df.loc[all_data_df['total frequency'] > min_common_words]
        # plot sankey diagram
        sk.make_sankey(all_data_df, 'title', 'words', label_color_dict=label_color_dict,
                       tricolor_colormap=tricolor_colormap,
                       vals='title frequency', pad=12, thickness=20)

    def plot_over_time(self, category, split_year=None, split=True, color_map=None, min_year=None, max_year=None):
        """
        plots a line graph of the change in a category variable over time (in years)
        :param category: string
            name of category that will be plotted on the y axis
        :param split_year: integer, optional
            before this year, everything plotted in one line
            after this year, plotting is split by label
        :param split: boolean
            determines whether data should be split by label; default is True
        :param color_map: dictionary, optional
            dictionary that maps labels to colors
        :param min_year: integer, optional
            minimum year included in plot
        :param max_year: integer, optional
            maximum year included in plot
        :return: None
        """
        # creates nested lists for x variable and y variable, separate sub-lists for different labels
        vals_dict = defaultdict(dict)

        for group in self.data.keys():
            vals_dict[group] = defaultdict(dict)
            # stores information about category and year (separately) mapped to title of file
            cat_dict = self.data[group][category]
            year_dict = self.data[group]["year"]
            for title_cat, val in cat_dict.items():
                for title_y, year, in year_dict.items():
                    if title_y == title_cat:
                        # populate dict with category value separated by group label
                        vals_dict[group][year] = val

        # checks if there will be a split in line graph by label at all
        if split:
            # checks if there is a starting year for the split
            if split_year:
                # creates dictionaries for before and after split
                pre_dict = defaultdict(dict)
                split_dict = defaultdict(dict)
                for group in vals_dict.keys():
                    split_dict[group] = defaultdict(dict)
                    for year, val in vals_dict[group].items():
                        # checks if text predates split year and adds information to appropriate dictionary
                        if year < split_year:
                            pre_dict[year] = val
                        else:
                            split_dict[group][year] = val
                # plots pre-split line
                pre_dict = OrderedDict(sorted(pre_dict.items()))
                plt.plot(pre_dict.keys(), pre_dict.values(), color="black")
                # plots post-split lines
                for group, group_dict in split_dict.items():
                    if len(group_dict) != 0:
                        group_dict = OrderedDict(sorted(group_dict.items()))
                        color, label = self._color_label(color_map, group)
                        plt.plot(group_dict.keys(), group_dict.values(), color=color, label=label)

            # if there is no split year
            else:
                # plots all years split by label
                for group, group_dict in vals_dict.items():
                    group_dict = OrderedDict(sorted(group_dict.items()))
                    color, label = self._color_label(color_map, group)
                    plt.plot(group_dict.keys(), group_dict.values(), color=color, label=label)
        # if no split
        else:
            # creates dictionary for all years
            combined_dict = defaultdict(dict)
            for group in vals_dict.keys():
                for year, val in vals_dict[group].items():
                    # populates dictionary with category value mapped to year
                    combined_dict[year] = val
            # plots single line for all years
            combined_dict = OrderedDict(sorted(combined_dict.items()))
            plt.plot(combined_dict.keys(), combined_dict.values(), color="black")

        plt.legend()
        plt.xlabel('Year')
        plt.ylabel(category.title())
        plt.title(category.title() + " by Year")
        plt.show()
