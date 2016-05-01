import cPickle

# Inputs: the file containing the input data, the length of the longest sentence
# Function: the input data is read in, the text is lower cased and split into words, and put into a datum format

def read_data_file(data_file,max_l):
    queries = []
    change_count = 0
    
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            line = line.lower()
            [label, text] = line.split('\t');
            newText = text.strip()
            words = newText.split()
            
             
            if len(words) > max_l:
                words = words[:max_l]
                change_count += 1
        
            datum = {"y": int(label),
                "text": " ".join(words),
                "num_words": len(words)}
                    
            queries.append(datum)
    return queries




def prep_test_data():
    test_file = './test/input_test_data.txt'
    output_test_file = './test/input_test.pkl'
     
    max_l = 100
    test_data = read_data_file(test_file, max_l)
    cPickle.dump(test_data, open(output_test_file, "wb"))




if __name__=="__main__":
    prep_test_data()
