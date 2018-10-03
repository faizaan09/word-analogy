import os,sys
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


model_path = './models/'
# loss_model = 'cross_entropy'
# loss_model = 'nce'
loss_model = sys.argv[1]

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""





def format_result(choices,res_pairs):
    choices = choices.split(',')

    res = choices + [choices[res_pairs[0]], choices[res_pairs[1]]]

    return ' '.join(res)



def get_best_pairs(ex_dists,ch_dists):
    most_ill = -1
    m_ind = 0
    least_ill = 1
    l_ind = 0


    for i in range(len(ch_dists)):
        
        cur_dist = sum([cosine(ch_dists[i],ex) for ex in ex_dists])/len(ex_dists)

        if cur_dist > most_ill:
            m_ind = i
            most_ill = cur_dist
        if cur_dist < least_ill:
            l_ind = i
            least_ill = cur_dist
    
    # print(l_ind, m_ind)
    return (l_ind, m_ind)




def get_distance(w1,w2):

    return abs(embeddings[dictionary[w2]] - embeddings[dictionary[w1]])


def parse_pairs(words):
    pairs = []
    
    for pair in words.split(','):
        pair = pair.strip('"').split(':')
        pairs.append(pair)

    return pairs

def main():
    pass
    result = []
    with open("word_analogy_dev.txt","r") as f:
        dev_set = f.read().split('\n')

    for sample in dev_set:
        examples, choices = sample.split("||")

        ex_dists = [get_distance(w1,w2) for w1,w2 in parse_pairs(examples)]
        ch_dists = [get_distance(w1,w2) for w1,w2 in parse_pairs(choices)]

        res_pairs = get_best_pairs(ex_dists,ch_dists)

        result.append(format_result(choices,res_pairs))

    with open("pred.txt",'w') as f:
        f.write('\n'.join(result))

if __name__ == '__main__':
    main()


def find_best_20():
    sim = model.similarity.eval()
    for i in xrange(valid_size):
    valid_word = reverse_dictionary[valid_examples[i]]
    top_k = 8  # number of nearest neighbors
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    log_str = "Nearest to %s:" % valid_word
    for k in xrange(top_k):
        close_word = reverse_dictionary[nearest[k]]
        log_str = "%s %s," % (log_str, close_word)