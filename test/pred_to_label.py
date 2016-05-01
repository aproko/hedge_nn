import sys

classifier = []

input = sys.argv[1]

# read in the predicted labels
with open('./test/input_results.pred', 'rU') as in1:
    for line in in1:
        classifier.append(int(line))

# read in the original input sentence
with open(input, 'rU') as in2:
    for line in in2:
        sentence = line

with open('./test/test_data_index.txt', 'rU') as in3:
    count = 0
    add = 0
    for line in in3:
        [index, hedge] = line.split(",")
        label = classifier[count]
        count = count + 1
        
        ind = int(index) - 12 + add #to account for the start symbols inserted in input_to_window.py
        new_str = sentence[0:ind] + " " + hedge.strip()+"_"+str(label) + " "  + sentence[ind + len(hedge) - 1:]
        sentence = new_str
        add = add + 2 #to account for the _1 or _0 that were just inserted

print sentence
        


