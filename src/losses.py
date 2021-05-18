import keras.backend as K

def hinge_loss_discriminator(y_true,y_pred):
    real_loss = K.mean(K.relu(1.0 - y_true))
    fake_loss = K.mean(K.relu(1.0 + y_pred))
    loss = real_loss + fake_loss
    return loss
def hinge_loss_generator(y_true,y_pred):
    fake_loss = -1*K.mean(y_pred)
    return fake_loss
def weighted_feature_matching_loss(y_true, fake_samples, image_input, real_samples, D, inner_weight, 
                          sample_weight):
    y_fake = D([image_input, fake_samples])[1:]
    y_real = D([image_input, real_samples])[1:]

    fm_loss = 0
    for i in range(len(y_fake)):
        if i<3:
            fm_loss += inner_weight[0] * K.mean(K.abs(y_fake[i] - y_real[i]))
        else:
            fm_loss += inner_weight[1] * K.mean(K.abs(y_fake[i] - y_real[i]))
    fm_loss *= sample_weight
    return fm_loss
