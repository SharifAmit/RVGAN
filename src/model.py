import tensorflow as tf
from keras.layers import Layer, InputSpec, Reshape, Activation, Conv2D, Conv2DTranspose, SeparableConv2D, Dropout
from keras.layers import Input, Add, Concatenate, Lambda,LeakyReLU,AveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomNormal
from functools import partial
from src.losses import *

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
def novel_residual_block(X_input, filters,base):

    name_base = base + '/branch'
    X = X_input
    X = ReflectionPadding2D((1,1),name=name_base + '1/rf')(X)
    X = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1),dilation_rate=1, padding='valid',name=name_base + '1/sepconv')(X)
    X = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1/BNorm')(X)
    X = LeakyReLU(alpha=0.2,name=name_base + '1/LeakyRelu')(X)

    ## Branch 1 ext1
    X_branch_1 = ReflectionPadding2D((1,1),name=name_base + '1_1/rf')(X)
    X_branch_1 = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), padding='valid',name=name_base + '1_1/sepconv')(X_branch_1)
    X_branch_1 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1_1/BNorm')(X_branch_1)
    X_branch_1 = LeakyReLU(alpha=0.2,name=name_base + '1_1/LeakyRelu')(X_branch_1)

    ## Branch 2
    X_branch_2 = ReflectionPadding2D((2,2),name=name_base + '2/rf')(X)
    X_branch_2 = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), dilation_rate=2, padding='valid',name=name_base + '2/sepconv')(X_branch_2)
    X_branch_2 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '2/BNorm')(X_branch_2)
    X_branch_2 = LeakyReLU(alpha=0.2,name=name_base + '2/LeakyRelu')(X_branch_2)
    X_add_branch_1_2 = Add(name=name_base + '1/add_branch1_2')([X_branch_2,X_branch_1])
    X = Add(name=name_base + '1/add_skip')([X_input, X_add_branch_1_2])
    return X

def disc_res_block(X_input, filters,base):

    name_base = base + '/branch'
    X = X_input

    ## Branch 1
    X_branch_1 = ReflectionPadding2D((1,1),name=name_base + '1_1/rf')(X)
    X_branch_1 = Conv2D(filters, kernel_size=(2,2), strides=(1,1), dilation_rate=2, padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name=name_base + '1_1/conv')(X_branch_1)
    X_branch_1 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1_1/BNorm')(X_branch_1)
    X_branch_1 = LeakyReLU(alpha=0.2,name=name_base + '1_1/LeakyRelu')(X_branch_1)

    ## Branch 2
    X_branch_2 = ReflectionPadding2D((1,1),name=name_base + '1_2/rf')(X)
    X_branch_2 = SeparableConv2D(filters, kernel_size=(2,2), strides=(1,1), dilation_rate=2, padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name=name_base + '1_2/sepconv')(X_branch_2)
    X_branch_2 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1_2/BNorm')(X_branch_2)
    X_branch_2 = LeakyReLU(alpha=0.2,name=name_base + '1_2/LeakyRelu')(X_branch_2)
    X_add_branch_1_2 = Add(name=name_base + '1_1/add_branch1_2')([X_branch_2,X_branch_1])

    X = Add(name=name_base + '2/add_skip')([X_input, X_add_branch_1_2])
    return X
    
def SFA(X,filters,i):
    X_input = X
    X = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="SFA_"+str(i+1)+"/conv1")(X)
    X = BatchNormalization(name="Attention_"+str(i+1)+"/BNorm1")(X)
    X = LeakyReLU(alpha=0.2,name="Attention_"+str(i+1)+"/leakyReLU1")(X)
    X = Add(name="Attention_"+str(i+1)+"/add1")([X_input,X])

    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="SFA_"+str(i+1)+"/conv2")(X)
    X = BatchNormalization(name="Attention_"+str(i+1)+"/BNorm2")(X)
    X = LeakyReLU(alpha=0.2,name="Attention_"+str(i+1)+"/leakyReLU2")(X)

    X = Add(name="Attention_"+str(i+1)+"/add2")([X_input,X])
    return X
def encoder_block(X,down_filters,i):
    X = Conv2D(down_filters, kernel_size=(4,4), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="down_conv_"+str(i+1))(X)
    X = BatchNormalization(name="down_bn_"+str(i+1))(X)
    X = LeakyReLU(alpha=0.2,name="down_leakyRelu_"+str(i+1))(X)
    return X

def decoder_block(X,up_filters,i):
    X = Conv2DTranspose(filters=up_filters, kernel_size=(4,4), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="up_convtranpose_"+str(i+1) )(X)
    X = BatchNormalization(name="up_bn_"+str(i+1))(X)
    X = LeakyReLU(alpha=0.2,name="up_leakyRelu_"+str(i+1))(X)
    return X

def coarse_generator(img_shape=(256, 256, 3),mask_shape=(256,256,1),ncf=64, n_downsampling=2, n_blocks=9, n_channels=1):
    X_input = Input(img_shape,name="input")
    X_mask  = Input(mask_shape,name="input_mask")
    X = Concatenate(axis=-1,name="concat")([X_input,X_mask])
    X = ReflectionPadding2D((3,3))(X)
    X = Conv2D(ncf, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="conv1")(X)
    X = BatchNormalization(name="bn_1")(X)
    X_pre_down = LeakyReLU(alpha=0.2,name="leakyRelu_1")(X)

    # Downsampling layers
    down_filters = ncf * pow(2,0) * 2
    X_down1 = encoder_block(X,down_filters,0)
    down_filters = ncf * pow(2,1) * 2
    X_down2 = encoder_block(X_down1,down_filters,1)
    X = X_down2


    # Novel Residual Blocks
    res_filters = pow(2,n_downsampling)
    for i in range(n_blocks):
        X = novel_residual_block(X, ncf*res_filters,base="block_"+str(i+1))


    # Upsampling layers
    up_filters  =int(ncf * pow(2,(n_downsampling - 0)) / 2) 
    X_up1 = decoder_block(X,up_filters,0)
    X_up1_att = SFA(X_down1,ncf*2,0)
    X_up1_add = Add(name="skip_1")([X_up1_att,X_up1])
    up_filters  =int(ncf * pow(2,(n_downsampling - 1)) / 2) 
    X_up2 = decoder_block(X_up1_add,up_filters,1)
    X_up2_att = SFA(X_pre_down,ncf,1)
    X_up2_add = Add(name="skip_2")([X_up2_att,X_up2])
    feature_out = X_up2_add
    print("X_feature",feature_out.shape)
    X = ReflectionPadding2D((3,3),name="final/rf")(X_up2_add)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="final/conv")(X)
    X = Activation('tanh',name="tanh")(X)


    model = Model(inputs=[X_input,X_mask], outputs=[X,feature_out],name='G_Coarse')
    model.compile(loss=['mse',None], optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model

def fine_generator(x_coarse_shape=(256,256,64),input_shape=(512, 512, 3), mask_shape=(512,512,1), nff=64, n_blocks=3, n_coarse_gen=1,n_channels = 1):

    
    X_input = Input(shape=input_shape,name="input")
    X_mask = Input(shape=mask_shape,name="input_mask")
    X_coarse = Input(shape=x_coarse_shape,name="x_input")
    print("X_coarse",X_coarse.shape)
    for i in range(1, n_coarse_gen+1):
        
        
        # Downsampling layers
        down_filters = nff * (2**(n_coarse_gen-i))
        X = Concatenate(axis=-1,name="concat")([X_input, X_mask])
        X = ReflectionPadding2D((3,3),name="rf_"+str(i))(X)
        X = Conv2D(down_filters, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="conv_"+str(i))(X)
        X = BatchNormalization(name="in_"+str(i))(X)
        X_pre_down = LeakyReLU(alpha=0.2,name="leakyRelu_"+str(i))(X)


        X_down1 = encoder_block(X,down_filters,i-1)
        # Connection from Coarse Generator
        X = Add(name="add_X_coarse")([X_coarse,X_down1])

        X = SeparableConv2D(down_filters*2, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="sepconv_"+str(i))(X)
        X = BatchNormalization(name="sep_in_"+str(i))(X)
        X = LeakyReLU(alpha=0.2,name="sep_leakyRelu_"+str(i))(X)
        for j in range(n_blocks-1):
            res_filters = nff * (2**(n_coarse_gen-i)) * 2
            X = novel_residual_block(X, res_filters,base="block_"+str(j+1))

        # Upsampling layers
        up_filters = nff * (2**(n_coarse_gen-i))
        X_up1 = decoder_block(X,up_filters,i-1)
        X_up1_att = SFA(X_pre_down,up_filters,i-1)
        X_up1_add = Add(name="skip_"+str(i))([X_up1_att,X_up1])

    X = ReflectionPadding2D((3,3),name="final/rf")(X_up1_add)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid',name="final/conv")(X)
    X = Activation('tanh',name="tanh")(X)

    model = Model(inputs=[X_input,X_mask,X_coarse], outputs=X, name='G_Fine')
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model


def discriminator_ae(input_shape_fundus=(512, 512, 3),
                        input_shape_label=(512, 512, 1),
                        ndf=32, n_layers=3, activation='tanh',
                        name='Discriminator'):
    X_input_fundus = Input(shape=input_shape_fundus,name="input_fundus")
    X_input_label = Input(shape=input_shape_label,name="input_label")

    features =[]
    X = Concatenate(axis=-1,name="concat")([X_input_fundus, X_input_label])
    
    X_down1 = encoder_block(X,ndf,0)
    X_down1_res = disc_res_block(X_down1, ndf,base="block1_down")
    features.append(X_down1_res)

    down_filters = ndf
    X_down2 = encoder_block(X_down1_res,down_filters,1)
    X_down2_res = disc_res_block(X_down2,down_filters,base="block2_down")
    features.append(X_down2_res)
    
    down_filters = ndf
    X_down3 = encoder_block(X_down2_res,down_filters,2)
    X_down3_res = disc_res_block(X_down3,down_filters,base="block3_down")
    features.append(X_down3_res)
    
    up_filters = ndf
    X_up1 = decoder_block(X_down3_res,up_filters,0)
    features.append(X_up1)

    X_up2 = decoder_block(X_up1,ndf,1)
    features.append(X_up2)

    X_up3 = decoder_block(X_up2,ndf,2)
    features.append(X_up3)


    X = Conv2D(1, kernel_size=(4,4), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X_up3)
    X = Activation(activation)(X)

    model = Model(inputs=[X_input_fundus, X_input_label], outputs=[X]+features, name=name)
    model.summary()
    model.compile(loss=['mse',None,None,None,None,None,None], optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))
    return model

def RVgan(g_model_fine,g_model_coarse, d_model1, d_model2,image_shape_fine,image_shape_coarse,image_shape_x_coarse,mask_shape_fine,mask_shape_coarse,label_shape_fine,label_shape_coarse,inner_weight):
    # Discriminator NOT trainable
    d_model1.trainable = False
    d_model2.trainable = False

    in_fine= Input(shape=image_shape_fine)
    in_coarse = Input(shape=image_shape_coarse)
    in_x_coarse = Input(shape=image_shape_x_coarse)
    in_fine_mask = Input(shape=mask_shape_fine)
    in_coarse_mask = Input(shape=mask_shape_coarse)
    label_fine = Input(shape=label_shape_fine)
    label_coarse = Input(shape=label_shape_coarse)

    # Generators
    gen_out_coarse, _ = g_model_coarse([in_coarse,in_coarse_mask])
    gen_out_fine = g_model_fine([in_fine,in_fine_mask,in_x_coarse])

    # Discriminators Fine
    dis_out_1_fake = d_model1([in_fine, gen_out_fine])

    # Discriminators Coarse
    dis_out_2_fake = d_model2([in_coarse, gen_out_coarse])

    #feature matching loss
    fm1 = partial(weighted_feature_matching_loss, image_input=in_fine,real_samples=label_fine, D=d_model1, inner_weight=inner_weight)
    #fm1 = partial(feature_matching_loss, image_input=in_fine,real_samples=label_fine, D=d_model1)
    fm2 = partial(weighted_feature_matching_loss, image_input=in_coarse,real_samples=label_coarse, D=d_model2, inner_weight=inner_weight)
    #fm2 = partial(feature_matching_loss, image_input=in_coarse,real_samples=label_coarse, D=d_model2)

    model = Model([in_fine,in_coarse,in_x_coarse,in_fine_mask,in_coarse_mask,label_fine,label_coarse], [dis_out_1_fake[0],
                                                    dis_out_2_fake[0],
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_coarse,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_fine
                                                    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['hinge', 
                    'hinge',
                    fm1,
                    fm2,
                    'hinge',
                    'hinge',
                    'mse',
                    'mse'
                    ], 
              optimizer=opt,loss_weights=[1,1,
                                          1,
                                          1,
                                          10,10,10,10
                                          ])
    model.summary()
    return model
