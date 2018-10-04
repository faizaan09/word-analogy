import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    B = tf.matmul(inputs,tf.transpose(true_w))
    A = tf.diag_part(B)
    B = tf.exp(B)
    B = tf.reduce_sum(B,1)

    return tf.subtract(tf.log(B), A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    sample: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

 
    unigram_prob = tf.convert_to_tensor(unigram_prob,dtype=tf.float32)
    sample = tf.convert_to_tensor(sample,dtype=tf.int32)

    target_w = tf.reshape(tf.gather(weights,labels),inputs.shape)
    target_b = tf.reshape(tf.gather(biases,labels),[-1,1])
    target_unigram_prob = tf.gather(unigram_prob,labels)


    main_PR =  tf.multiply(inputs,target_w)
    main_PR = tf.reduce_sum( main_PR, 1, keepdims=True)
    main_PR = tf.add(main_PR,target_b)


    log_prob = tf.log(sample.shape[0].value*target_unigram_prob + 1e-10)
    main_PR = tf.log(tf.sigmoid(main_PR - log_prob)+ 1e-10)

    sample_weights = tf.reshape(tf.gather(weights,sample),[sample.shape[0],inputs.shape[1]])
    sample_biases = tf.reshape(tf.gather(biases,sample),[-1,])
    sample_unigram_prob = tf.gather(unigram_prob,sample)


    sample_PR =  tf.matmul(inputs,tf.transpose(sample_weights))
    sample_PR = tf.add(sample_PR,sample_biases)

    log_prob = tf.log(sample.shape[0].value*sample_unigram_prob+ 1e-10)
    sample_PR = tf.sigmoid(sample_PR - log_prob)
    sample_PR = tf.log(1-sample_PR + 1e-10)
    sample_PR = tf.reduce_sum((sample_PR),1)
    
    ans = tf.subtract(-main_PR,sample_PR)
    
    return ans










