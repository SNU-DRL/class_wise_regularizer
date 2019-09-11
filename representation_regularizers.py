import tensorflow as tf


def get_cr_loss(act, is_normalized=False, num_units=0):
    '''
        Decov (CR) loss described in 'Reducing Overfitting In Deep Networks by Decorrelating Representation', ICLR 2016
        :param act: mini-batch activations
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :return: sum of covariance of all units
    '''

    act_mean = tf.reduce_mean(act, 0, True)
    act_centralized = tf.expand_dims(act - act_mean, 2)
    corr = tf.reduce_mean(tf.matmul(act_centralized, tf.transpose(act_centralized, perm=[0, 2, 1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag = tf.diag_part(corr)
    corr_diag = tf.matrix_diag_part(corr)
    corr_diag_sqr = tf.square(corr_diag)
    corr_diag_sqr_sum = tf.reduce_sum(corr_diag_sqr)
    loss = 0.5 * (corr_frob_sqr - corr_diag_sqr_sum)

    if is_normalized:
        normalization_factor = num_units * (num_units - 1) * 0.5

        return tf.scalar_mul(1.0 / normalization_factor, loss)
    else:
        return loss


def get_cw_cr_loss(act, batch_labels, num_labels, is_normalized=False, num_units=0, is_onehot=True):
    '''
        cw-CR loss described in 'Utilizing Class Information for Deep Network Representation Shaping', AAAI 2019
        :param act: mini-batch activations
        :param batch_labels: labels of mini-batch samples
        :param num_labels: the number of labels (classes)
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :param is_onehot: whether labels are onehot or not
        :return: sum of covariance of all units per class
    '''

    if is_onehot:
        # one hot to number, e.g., [[0, 0, 1], [1, 0, 0]] -> [2, 0]
        batch_labels = tf.argmax(tf.transpose(batch_labels), axis=0)

    sum_cross_covariances = 0.0
    for label in range(num_labels):
        label_act = tf.gather_nd(act, tf.where(tf.equal(batch_labels, label)))  # select activation with 'label'

        act_mean = tf.reduce_mean(label_act, 0, True)
        act_centralized = tf.expand_dims(label_act - act_mean, 2)
        corr = tf.reduce_mean(tf.matmul(act_centralized, tf.transpose(act_centralized, perm=[0, 2, 1])), 0)

        corr_frob_sqr = tf.reduce_sum(tf.square(corr))
        corr_diag = tf.diag_part(corr)
        corr_diag = tf.matrix_diag_part(corr)
        corr_diag_sqr = tf.square(corr_diag)
        corr_diag_sqr_sum = tf.reduce_sum(corr_diag_sqr)

        label_cross_covariance = corr_frob_sqr - corr_diag_sqr_sum
        sum_cross_covariances += label_cross_covariance

    if is_normalized:
        normalization_factor = num_units * (num_units - 1) * 0.5 * num_labels

        return tf.scalar_mul(1. / normalization_factor, 0.5 * sum_cross_covariances)
    else:
        return 0.5 * sum_cross_covariances


def get_vr_loss(act, is_normalized=False, num_units=0):
    '''
        VR loss described in 'Utilizing Class Information for Deep Network Representation Shaping', AAAI 2019
        :param act: mini-batch activations
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :return: sum of variance of all units
    '''

    _, variances = tf.nn.moments(act, axes=[0])

    if is_normalized:
        return tf.scalar_mul(1.0 / num_units, tf.reduce_sum(variances))
    else:
        return tf.reduce_sum(variances)


def get_cw_vr_loss(act, batch_labels, num_labels, is_normalized=False, num_units=0, is_onehot=True):
    '''
        cw-VR loss described in 'Utilizing Class Information for Deep Network Representation Shaping', AAAI 2019
        :param act: mini-batch activations
        :param batch_labels: labels of mini-batch samples
        :param num_labels: the number of labels (classes)
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :param is_onehot: whether labels are onehot or not
        :return: sum of variance of all units per class
    '''

    if is_onehot:
        # one hot to number, e.g., [[0, 0, 1], [1, 0, 0]] -> [2, 0]
        batch_labels = tf.argmax(tf.transpose(batch_labels), axis=0)

    layer_variances = []
    for label in range(num_labels):
        label_act = tf.gather_nd(act, tf.where(tf.equal(batch_labels, label)))  # select activation with 'label'
        label_means, label_variances = tf.nn.moments(label_act, axes=[0])
        layer_variances = tf.concat([layer_variances, [tf.reduce_sum(label_variances)]], 0)

    if is_normalized:
        normalization_factor = num_units * num_labels

        return tf.scalar_mul(1. / normalization_factor, tf.reduce_sum(layer_variances))
    else:
        return tf.reduce_sum(layer_variances)


def get_l1r_loss(act, is_normalized=False, num_units=0):
    '''
        L1R
        :param act: mini-batch activations
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :return: sum of absolute amplitude of activation
    '''
    if is_normalized:
        return tf.scalar_mul(1.0 / num_units, tf.reduce_sum(tf.abs(act)))
    else:
        return tf.reduce_sum(tf.abs(act))


def get_rr_loss(act, is_normalized=False, num_units=0):
    '''
        RR loss proposed in this work
        :param act: mini-batch activations
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :return: approximate stable rank of an activation matrix
    '''

    corr = tf.matmul(tf.transpose(act), act)
    col_max = tf.reduce_max(tf.reduce_sum(tf.abs(act), 0))
    row_max = tf.reduce_max(tf.reduce_sum(tf.abs(act), 1))
    corr_diag = tf.diag_part(corr)
    corr_diag_sum = tf.reduce_sum(corr_diag)
    loss = corr_diag_sum / (col_max * row_max)

    if is_normalized:
        return tf.scalar_mul(1.0 / num_units, loss)
    else:
        return loss

    return appr_rank


def get_cw_rr_loss(act, batch_labels, num_labels, is_normalized=False, num_units=0, is_onehot=True):
    '''
        cw-RR loss proposed in this work
        :param act: mini-batch activations
        :param batch_labels: labels of mini-batch samples
        :param num_labels: the number of labels (classes)
        :param is_normalized: whether normalized or not
        :param num_units: the number of units
        :param is_onehot: whether labels are onehot or not
        :return: sum of approximate stable ranks of an activation matrix per class
    '''

    if is_onehot:
        # one hot to number, e.g., [[0, 0, 1], [1, 0, 0]] -> [2, 0]
        batch_labels = tf.argmax(tf.transpose(batch_labels), axis=0)

    total_loss = 0
    for label in range(num_labels):
        label_act = tf.gather_nd(act, tf.where(tf.equal(batch_labels, label)))  # select activation with 'label'

        corr = tf.matmul(tf.transpose(label_act), label_act)
        col_max = tf.reduce_max(tf.reduce_sum(tf.abs(label_act), 0))
        row_max = tf.reduce_max(tf.reduce_sum(tf.abs(label_act), 1))
        corr_diag = tf.diag_part(corr)
        corr_diag_sum = tf.reduce_sum(corr_diag)
        loss = corr_diag_sum / (col_max * row_max)

        total_loss += loss

    if is_normalized:
        return tf.scalar_mul(1.0 / (num_units * num_labels), total_loss)
    else:
        return loss
