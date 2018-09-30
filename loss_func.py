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
    target_b = tf.reshape(tf.gather(biases,labels),[-1,])


    all_PR =  tf.matmul(inputs,tf.transpose(target_w))
    all_PR = tf.add(all_PR,target_b)

    main_PR = tf.diag_part(all_PR)

    target_unigram_prob = tf.gather(unigram_prob,labels)
    # print(type(sample))
    log_prob = tf.log(sample.shape[0].value*target_unigram_prob + 1e-9)

    main_PR = tf.log(tf.sigmoid(main_PR - log_prob)+ 1e-9)

    # main_PR = tf.Print(main_PR,[main_PR],message="main_PR is...",first_n=2)

    sample_weights = tf.reshape(tf.gather(weights,sample),[sample.shape[0],inputs.shape[1]])
    sample_biases = tf.reshape(tf.gather(biases,sample),[-1,])


    sample_PR =  tf.matmul(inputs,tf.transpose(sample_weights))
    sample_PR = tf.add(sample_PR,sample_biases)


    sample_unigram_prob = tf.gather(unigram_prob,sample)

    log_prob = tf.log(sample.shape[0].value*sample_unigram_prob+ 1e-9)

    # log_prob = tf.Print(log_prob,[log_prob],message="log_prob is...",first_n=2)

    sample_PR = tf.sigmoid(sample_PR - log_prob)

    # sample_PR = tf.Print(sample_PR,[sample_PR],message="tf.sigmoid(sample_PR - log_prob)...",first_n=2)

    sample_PR = tf.log(1-sample_PR + 1e-9)

    # sample_PR = tf.Print(sample_PR,[sample_PR],message="tf.log(1-sample_PR)...",first_n=2)

    sample_PR = tf.reduce_sum((sample_PR),1)
    
    # sample_PR = tf.Print(sample_PR,[sample_PR],message="sample_PR is...",first_n=2)

    ans = tf.subtract(-main_PR,sample_PR)

    # ans = tf.Print(ans,[ans],message="Ans is...",first_n=2)
    # sess = tf.Session()
    # print(sess.run(ans))
    return ans










