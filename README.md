# word-analogy

Python Version: 3.6.4

Baseline model Accuracy: 
 1. NCE :  
 2. Cross Entropy loss:

Best model:
 1. NCE 
 
    | neg_samples | Batch_size | Emb_size | Skip_window | num_skips | Learning_rate | Epochs  | Accuracy |
    |-------------|------------|----------|-------------|-----------|---------------|---------|----------|
    | 64          | 128        | 128      | 5           | 8         | 0.001         | 1000000 | 33.2%    |

 2. Cross Entropy loss 

    | Batch_size | Emb_size | Skip_window | num_skips | Learning_rate | Epochs  | Accuracy |
    |------------|----------|-------------|-----------|---------------|---------|----------|
    | 128        | 128      | 8           | 8         | 0.005         | 400000  | 33.6%    |


## Implementation details:

### Batch generation:

We consider a particular window size (2 * skip_window + 1) that consists all the context words and our target word.
Sample 'num_skips' wordss from the context word window
Add every (target_word, context_word) pair from the current window till we generate enough samples for the batch
If we exhaust the current window before we have enough samples for the batch, we shift the window by an index
If we reach the end of the dataset, we loop over it.


### Cross Entropy Loss

* Perform matrix multiplication of input matrix of shape [batch_size, embedding_size] and transpose of the weight vector of shape [batch_size, embedding_size].
* As a result of this matrix multiplication we get a matrix ('B') of shape [batch_size,batch_size]. 
* We extract the diagonal elements of this matrix that form 'A' for our implementation (we extract A before the exp operation because taking the log will nullify that operation)
* Now we take the exp of every element of B, take a sum over all columns and form a column vector of size 'batch_size'
* We return the difference between the log of elements of B and elements of A, as this would be the negative log-likelihood we are trying to calculate

### NCE loss

* Multiply the input with the weights for the particular labels and add bias to it
* Subtract log of k times the unigram probability of the target word from the previous result and take the sigmoid of the new result (A)
* Repeat the above steps replacing the input of target words with predecided negative samples for NCE (B)
* Take a sum over all columns of B to get one negative sample sum for every row in the batch.
* Take the negative of the sum of A and B, this is the value we want to return as our loss
* side note: while taking logs, add a small number like 1e-10 to prevent the loss function from exploding


### General Notes

* The word2vec_basic.py has been modified to take certain parameters as input from config.txt (where every line has hyperparameter details)
* The file then goes on to train a model, run it on the word_analogy task, score the results and save the results in a separate file results.txt
* word_analogy.py currently writes all its outputs to pred.txt, it can be parameterised if needed.
* finding top 20 similar words to [first, american, would] is implemented as a function in word_analogy.py

## Experiment Details

| Loss          | neg_samples | Batch_size | Emb_size | Skip_window | num_skips | Learning_rate | Epochs  | Accuracy |
|---------------|-------------|------------|----------|-------------|-----------|---------------|---------|----------|
| nce           | 64          | 128        | 128      | 5           | 8         | 0.01          | 200001  | 34.6%    |
| nce           | 64          | 128        | 128      | 5           | 8         | 0.001         | 1000001 | 28.3%    |
| nce           | 64          | 128        | 128      | 8           | 8         | 0.01          | 200001  | 30.2%    |
| nce           | 64          | 128        | 128      | 8           | 8         | 0.001         | 1000001 | 32.0%    |
| nce           | 64          | 128        | 256      | 5           | 8         | 0.01          | 200001  | 32.7%    |
| nce           | 64          | 128        | 256      | 5           | 8         | 0.001         | 1000001 | 31.7%    |
| nce           | 64          | 128        | 256      | 8           | 8         | 0.01          | 200001  | 31.9%    |
| nce           | 64          | 128        | 256      | 8           | 8         | 0.001         | 1000001 | 30.4%    |
| nce           | 128         | 128        | 128      | 5           | 8         | 0.01          | 200001  | 29.8%    |
| nce           | 32          | 128        | 128      | 5           | 8         | 0.01          | 200001  | 31.9%    |
| nce           | 64          | 128        | 128      | 5           | 8         | 0.01          | 140001  | 33.4%    |
| nce           | 64          | 128        | 128      | 5           | 8         | 0.01          | 200000  | 33.4%    |
| nce           | 64          | 128        | 128      | 5           | 8         | 0.001         | 1000000 | 33.2%    |
| nce           | 64          | 256        | 128      | 5           | 8         | 0.01          | 200000  | 33.4%    |
| cross_entropy |             | 128        | 128      | 5           | 8         | 0.01          | 200001  | 33.4%    |
| cross_entropy |             | 128        | 128      | 8           | 8         | 0.01          | 200001  | 32.1%    |
| cross_entropy |             | 128        | 128      | 8           | 8         | 0.005         | 400001  | 32.5%    |
| cross_entropy |             | 128        | 256      | 5           | 8         | 0.01          | 200001  | 31.0%    |
| cross_entropy |             | 256        | 256      | 5           | 8         | 0.005         | 400001  | 29.9%    |
| cross_entropy |             | 128        | 256      | 8           | 8         | 0.01          | 200001  | 31.7%    |
| cross_entropy |             | 128        | 256      | 8           | 8         | 0.005         | 400001  | 31.8%    |
| cross_entropy |             | 256        | 128      | 5           | 8         | 0.01          | 200000  | 33.4%    |
| cross_entropy |             | 128        | 128      | 8           | 8         | 0.005         | 400000  | 33.6%    |



