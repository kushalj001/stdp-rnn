import pandas as pd
from collections import Counter
import torch
import numpy as np
import random

def load_data(path):
    """
    Loads the text data, i.e. questions and labels from the text file.
    Returns list of questions, fine labels and coarse labels.
    Note that we do not use fine labels currently.
    """
    with open(path, "r", encoding="ISO-8859-1") as f:
        lines = f.readlines()
        
    questions = []
    coarse_labels = []
    fine_labels = []
    for line in lines:
        tokens = line.split()
        question = " ".join([tok for tok in tokens[1:]])
        label = tokens[0].split(":")
        coarse_labels.append(label[0])
        fine_labels.append(label[1])
        questions.append(question)
    return questions, coarse_labels, fine_labels


def normalize_spaces(text):
    """
    Removes extra white spaces from the text.
    """
    import re
    text = re.sub(r'\s', ' ', text)
    return text

def convert_to_lower(text):
    """
    Converts the text to lowercase. This is important because
    we're currently using glove static embeddings to embed each word.
    Glove has embeddings for lowercase words only.
    """
    text = " ".join([word.lower() for word in text.split()])
    return text

def preprocess(questions):
    """
    Perform the text preprocessing steps on each question.
    """
    clean_questions = []
    for qtn in questions:
        text = normalize_spaces(qtn)
        text = convert_to_lower(text)
        clean_questions.append(text)
    return clean_questions


def build_word_vocab(vocab_text):
    '''
    Builds a word-level vocabulary from the given text.
    
    :param list vocab_text: list of questions
    :returns 
        dict word2idx: word to index mapping of words
        dict idx2word: integer to word mapping
        list word_vocab: list of words sorted by frequency
    '''
    words = []
    for sent in vocab_text:
        for word in sent.split():
            words.append(word)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word:idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v:k for k,v in word2idx.items()}
    
    
    return word2idx, idx2word, word_vocab

def convert_to_dataframe(questions, coarse_labels):
    """
    Converts the dataset into a dataframe for easy access of 
    input questions and the corresponding labels.
    """
    data_list = []
    for i in range(len(questions)):
        data_list.append({"question":questions[i], "label_text": coarse_labels[i]})
        
    data = pd.DataFrame(data_list)
    data.label_text = pd.Categorical(data.label_text)
    data["label"] = data.label_text.cat.codes
    
    return data

def question_to_ids(questions, word2idx):
    """
    Converts question text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using whitespace tokenizer first.
    
    :param list questions: list of question text to be converted
    :param dict word2idx: word to id mapping
    :returns list context_ids: list of mapped ids
    
    :raises assertion error: sanity check
    
    """
    qtn_ids = []
    for i, text in enumerate(questions):
        qtn_tokens = text.split()
        qtn_id = [word2idx[word] for word in qtn_tokens]
        qtn_ids.append(qtn_id)
    
    return qtn_ids


class QuestionClassificationDataLoader:
    
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # word ids of questions
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
    
        for batch in self.data:
            max_qtn_len = max([len(qtn) for qtn in batch.question_ids])
            padded_qtn = torch.LongTensor(len(batch), max_qtn_len).fill_(1)
            
            for i, qtn in enumerate(batch.question_ids):
                padded_qtn[i, :len(qtn)] = torch.LongTensor(qtn)
            
            label = torch.LongTensor(list(batch.label))
            
            yield {"questions":padded_qtn, "labels":label}

## Only a single function is needed to generate data for copy task
# TODO: setup data generation for incremental learning setup
def copy_task_dataloader(num_batches, batch_size, seq_width, min_seq_len, max_seq_len):
    for batch_num in range(num_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_seq_len, max_seq_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        yield batch_num+1, inp.float(), outp.float()
