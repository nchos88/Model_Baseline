from keras import layers , models 

#inputShape =(52,52,1)

def BatchAct(x):
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x


# shortcut model is inspired by Identity Mappings in Deep Residual Networks paper
# you can use grouped convolution from ResNext
# grouped_convolution code is from https://github.com/damgambit/ccn-statoil-iceberg-classifier/blob/master/ResNet.ipynb
def grouped_convolution(y, nb_channels, _strides , cardinality):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y


def ShortCut(x , channel1 , channel2 ):
    origin = x
    branch = BatchAct(x)
    branch = layers.Conv2D(channel1 , padding = 'same')(branch)
    branch = BatchAct(branch)
    branch = layers.Conv2D(channel2 , padding = 'same')(branch)
    output = layers.add([branch , x])
    return output

def ResNet_v1(ly_input, c):
    origin = layers.Conv2D(64 , kernel_size=(7,7) , strides = (2,2) , padding = 'same')(input)
    origin = BatchAct(origin)

    origin = layers.MaxPool2D( pool_size=(3,3) , strides=(2,2) , padding ='same' )(origin)
    origin = ShortCut(origin , 128, 256)
    origin = ShortCut(origin , 256, 512)
    origin = ShortCut(origin , 512, 1024)
    origin = ShortCut(origin , 1024,2048)

    feature = layers.GlobalAveragePooling2D()(origin)
    output = layers.Dense(nb_output)
    return output



    

