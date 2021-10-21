import keras.backend as K


def weighted_feature_matching_loss(y_true, y_pred):
    fm_loss = 0
    for i in range(len(y_true)):
        fm_loss += 0.5*K.mean(K.abs(y_true[i] - y_pred[i]))
    return fm_loss
