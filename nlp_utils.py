import csv
import os
from wildnlp.aspects import qwerty, articles, swap, sentiment_masking, remove_char
from wildnlp.datasets.base import Dataset 
import numpy as np
import torch
import random

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
    weights_matrix = np.random.randn(len(word_vocab), 100)
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            not_found.append(word)
    return weights_matrix, not_found


def text_to_ids(examples, word2idx):
    """
    Converts question text to their respective ids by mapping each word
    using word2idx. Input text is tokenized using whitespace tokenizer first.
    
    :param list questions: list of question text to be converted
    :param dict word2idx: word to id mapping
    :returns list context_ids: list of mapped ids
    
    :raises assertion error: sanity check
    
    """
    text_ids = []
    for i, ex in enumerate(examples):
        text_tokens = ex["content"].split()
        text_id = [word2idx.get(word, word2idx["<unk>"]) for word in text_tokens]
        text_ids.append(text_id)
    
    return text_ids




class IMDB(Dataset):
    """The IMDB dataset containing movie reviews for a sentiment analysis.
    The dataset consists of 50 000 reviews of two classes, negative and positive.
    Each review is stored in a separate text file.
    For details see: http://ai.stanford.edu/~amaas/data/sentiment/
    """

    def load(self, path):
        """Loads a SNLI dataset.

        :param path: A path to single file, directory containing review files
                     or list of paths to such directories.

        :return: None
        """

        if type(path) is str and os.path.isdir(path):
            self._load_multiple_files(path)

        elif type(path) is list:
            for single_path in path:
                self._load_multiple_files(single_path)

        elif os.path.isfile(path):
            _, filename = os.path.split(path)
            entry = {'path': filename,
                     'content': self._read_file(path)}
            self._data.append(entry)

    def apply(self, aspect):
        """Modifies contents of the whole files in the IMDB dataset.
        """

        modified = []
        for entry in self._data:
            modified_sentence = aspect(entry['content'])
            entry['content'] = modified_sentence
            modified.append(entry)

        return modified

    def save(self, data, path):
        """Saves IMDB reviews to separate files
        with the original names.

        :param path: path to a top directory where files will be saved.

        :return: None
        """

        for entry in data:
            directory, filename = os.path.split(entry['path'])
            full_path = os.path.join(path, directory)
            if not os.path.exists(full_path) and directory != '':
                os.makedirs(full_path)
            with open(os.path.join(path, entry['path']), 'w') as f:
                f.write(entry['content'])

    def save_tsv(self, data, path):
        """Convenience function for saving IMDB reviews into a single TSV file.

        :param path: Path to a tab separated file.

        :return: None
        """

        with open(path, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(['Sentiment', 'Content'])

            for entry in data:
                directory, _ = os.path.split(entry['path'])
                if directory == 'neg':
                    sentiment = 'neg'
                elif directory == 'pos':
                    sentiment = 'pos'
                else:
                    sentiment = 'unsup'

                writer.writerow([sentiment, entry['content']])

    @staticmethod
    def _read_file(path):

        with open(path, 'r',encoding="utf-8") as f:
            content = f.read()

        return content

    def _load_multiple_files(self, path):

        filenames = os.listdir(path)
        for filename in sorted(filenames):
            full_path = os.path.join(path, filename)

            _, patent_dir = os.path.split(path)
            entry = {'path': os.path.join(patent_dir, filename),
                     'content': self._read_file(full_path)}
            self._data.append(entry)

def convert_to_lower(text):
    """
    Converts the text to lowercase. This is important because
    we're currently using glove static embeddings to embed each word.
    Glove has embeddings for lowercase words only.
    """
    text = " ".join([word.lower() for word in text.split()])
    return text

def load_data(dir_path):
    dataset = IMDB()
    dataset.load(dir_path+"neg")
    for ex in dataset.data:
        ex["label"] = 0
        ex["content"] = convert_to_lower(ex["content"])
    dataset.load(dir_path+"pos")
    for ex in dataset.data:
        if "label" not in ex:
            ex["label"] = 1
            ex["content"] = convert_to_lower(ex["content"])
    return dataset


def build_word_vocab(vocab_text):
    '''
    Builds a word-level vocabulary from the given text.
    
    :param list vocab_text: list of questions
    :returns 
        dict word2idx: word to index mapping of words
        dict idx2word: integer to word mapping
        list word_vocab: list of words sorted by frequency
    '''
    from collections import Counter
    words = []
    for sent in vocab_text:
        for word in sent.split():
            words.append(word)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    word_vocab = word_vocab[:25_000]
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word:idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v:k for k,v in word2idx.items()}
    
    
    return word2idx, idx2word, word_vocab


def convert_to_dataframe(examples):
    """
    Converts the dataset into a dataframe for easy access of 
    input questions and the corresponding labels.
    """
    import pandas as pd
    data_list = []
    for i in range(len(examples)):
        data_list.append({"review":examples[i]["content"], "label": examples[i]["label"]})
    
    data = pd.DataFrame(data_list)
    return data


class IMDBDataloader:
    
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        #data = shuffle(data, random_state=random.seed(1234))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        random.shuffle(data)
        # word ids of questions
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
    
        for batch in self.data:
            batch["len"] = batch["text_ids"].str.len()
            batch = batch.sort_values(by="len", ascending=False).drop(columns="len")
            
            txt_lengths = [len(txt) for txt in batch.text_ids]
            max_txt_len = txt_lengths[0]
            padded_txt = torch.LongTensor(len(batch), max_txt_len).fill_(1)
            #sorted_ids = list(batch.text_ids)
            #sorted_ids.sort(key=lambda item: -len(item))
            for i, txt in enumerate(batch.text_ids):
                padded_txt[i, :len(txt)] = torch.LongTensor(txt)
            
            label = torch.LongTensor(list(batch.label))
            txt_lengths = torch.LongTensor(txt_lengths)
            yield {"text":padded_txt, "labels":label, "text_lengths":txt_lengths}


def corrupt_qwerty(dataset, perturb_percent):
    dataset.apply(qwerty.QWERTY(words_percentage=perturb_percent, characters_percentage=perturb_percent))
    return dataset

def corrupt_sentiment_masking(dataset):
    dataset.apply(sentiment_masking.SentimentMasking())
    return dataset

def corrupt_remove_char(dataset, perturb_percent):
    dataset.apply(remove_char.RemoveChar(words_percentage=perturb_percent, characters_percentage=perturb_percent))
    return dataset


def get_df(dataset, word2idx):
    df = convert_to_dataframe(dataset.data)
    text_ids = text_to_ids(dataset.data, word2idx)
    df["text_ids"] = text_ids
    return df

