from nlp_library import Text, map_parts_speech
import pprint as pp
import os
import regex as re


def read_file(filename):
    data = {}
    with open(filename, "r") as infile:
        file_data = infile.readlines()
        for line in file_data[1:]:
            line = line.strip().split("\t")
            data[int(line[0])] = line[-1]
    return data


def read_directory_files(directory):
    # assign directory

    # iterate over files in
    # that directory
    paths = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            paths.append(f)
    return paths


def main():
    party_by_year = read_file("president_affiliation.txt")

    paths = read_directory_files("sotu")
    tt = Text()
    for path in paths:
        tt.load_text(
            path,
            year=int(re.findall('\d+', path)[0]),
            label=party_by_year[int(re.findall('\d+', path)[0])],
            title=path.split(".")[0].replace("_", " ").replace("sotu/", "")
        )
    tt.frequency_filter(10, "word count")
    tt.frequency_filter(10, "parts of speech")
    tt.rename_keys("parts of speech",
                   map_parts_speech("Parts_of_Speech.txt")
    )
    tt.time_word_cloud(20, 1939, 2020, ["Republican", "Democrat"])
    tt.sankey_diagram(min_common_words=500,
                      label_color_dict={'Republican': (255, 0, 0),
                                        'Democrat': (0, 0, 255)
                      },
                      min_year=1936
    )
    color_map = {"Democrat": "blue",
                 "Republican": "red"
    }
    tt.plot_over_time("readability difficulty",
                      split_year=1936,
                      color_map=color_map
    )


if __name__ == '__main__':
    main() 
