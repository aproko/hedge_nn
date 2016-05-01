import csv
import string
import collections
import sys

multi_word_dict = []
single_word_dict = []

index_to_string = {} #int index, string words[-2,+2] around hedgeword
index_to_hedge = {} #int index, string hedgeword

# Inputs: the sentence and the dictionary from which we take the potential hedge words.
# Function: The sentence is padded with two start symbols and 2 stop symbols to account for hedges that appear at the start of end of the sentence. Then a window of 2 words on each side of the potential hedge word (or phrase) is selected.

def find_hedges(sentence, dictionary):
    new_str = "start start " + sentence.translate(string.maketrans("",""), string.punctuation) + " stop stop"
    for itm in dictionary:
        item = " " + itm + " "
        if item in new_str:
            start_index = new_str.index(item)
            index_to_hedge[start_index] = item
            before = new_str[:start_index]
            after = new_str[start_index+len(item):]
            before_words = before.strip().split()
            after_words = after.strip().split()
            window = before_words[-2] + " " + before_words[-1]  + item + after_words[0] + " " + after_words[1]
            index_to_string[start_index] = window

# Inputs: the file from which to read the hedge word dictionary, mode (either m for multi-word, or s for single-word)
# Function: The hedge word dictionary is read in and stored in either multi_word_dict (with punctuation removed), or single_word_dict

def read_dict(filename, mode):
    with open(filename, 'rU') as input:
        reader = csv.reader(input)
        for line in reader:
            hedge = line[0]
            if "m" in mode:
                multi_word_dict.append(hedge.translate(string.maketrans("",""), string.punctuation))
            else:
                single_word_dict.append(hedge)



if __name__=="__main__":
    
    input_file_path = sys.argv[1]
    
    read_dict('multiword_dict.csv', "m")
    read_dict('dictionary.csv', "d")


    with open(input_file_path, 'rU') as infile:
        for line in infile:
            find_hedges(line, multi_word_dict)
            find_hedges(line, single_word_dict)

    sorted_list = collections.OrderedDict(sorted(index_to_string.items()))
    sorted_dict = collections.OrderedDict(sorted(index_to_hedge.items()))

# Both input_test_data.txt and test_data_index.txt are sorted by the character index of the potential hedge word in ascending order.

    with open('./test/input_test_data.txt', 'w') as data_out:
        for key, val in sorted_list.iteritems():
            data_out.write("0\t"+val+"\n")

    with open('./test/test_data_index.txt', 'w') as index_out:
        for i,v in sorted_dict.iteritems():
            w_str = str(i) + "," + v + "\n"
            index_out.write(w_str)
