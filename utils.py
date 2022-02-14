import numpy as np



def create_glove_matrix():
    '''
    Parses the glove word vectors text file and returns a dictionary with the words as
    keys and their respective pretrained word vectors as values.

    '''
    glove_dict = {}
    with open("glove.6B.100d.txt/glove.6B.100d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove_dict[word] = vector

    f.close()
    
    return glove_dict


def create_word_embedding(glove_dict, word_vocab):
    '''
    Creates a weight matrix of the words that are common in the GloVe vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    not_found = []
    weights_matrix = np.zeros((len(word_vocab), 100))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            not_found.append(word)
    return weights_matrix, not_found


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs