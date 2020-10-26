import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Multiply, Conv2DTranspose, Input, BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Multiply, Add, Reshape
from tensorflow.keras import backend as K


class Conv2dBn(tf.keras.Model):
    def __init__(self, input_shape, filters, kernel_size, padding='same', strides=1, activation='relu', **kwargs):
        super(Conv2dBn, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(Conv2dBn, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs
        )

    def model(self):
        return Model(inputs=self.input_layer, outputs=self.output_layer)

    def summary(self, line_length=None, positions=None, print_fn=None):
        model = Model(inputs=self.input_layer, outputs=self.output_layer)
        return model.summary()

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer,
        )

    def call(self, inputs, training=False):
        x = Conv2D(self.filters, self.kernel_size, kernel_initializer='he_normal', padding=self.padding, strides=self.strides)(inputs)
        x = BatchNormalization()(x)
        if self.activation:
            x = Activation(self.activation)(x)

        return x


class SeBlock(tf.keras.Model):
    def __init__(self, input_shape, reduction_ratio=16, **kwargs):
        super(SeBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.input_layer = Input(shape=input_shape)
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(SeBlock, self).__init__(
            self.input_layer,
            self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs, training=False):
        ch_input = K.int_shape(inputs)[-1]
        ch_reduced = ch_input // self.reduction_ratio

        # Squeeze
        x = GlobalAveragePooling2D()(inputs)

        # Excitation
        x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x)  # Eqn.3
        x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x)  # Eqn.3

        x = Reshape((1, 1, ch_input))(x)
        x = Multiply()([inputs, x])

        return x


class SeResidualBlock(tf.keras.Model):
    def __init__(self, input_shape, filter_sizes, strides=1, reduction_ratio=16, **kwargs):
        super(SeResidualBlock, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape)
        self.filter_1, self.filter_2, self.filter_3 = filter_sizes
        self.strides = strides
        self.reduction_ratio = reduction_ratio

        self.conv1 = Conv2dBn(input_shape, self.filter_1, (1, 1), strides=self.strides)
        self.conv2 = Conv2dBn(self.conv1.output_shape_no_batch, self.filter_2, (3, 3))
        self.conv3 = Conv2dBn(self.conv2.output_shape_no_batch, self.filter_3, (1, 1), activation=None)
        self.seBlock = SeBlock(self.conv3.output_shape_no_batch, self.reduction_ratio)
        self.projectedInput = Conv2dBn(input_shape, self.filter_3, (1, 1), strides=self.strides, activation=None)
        self.output_layer = self.call(self.input_layer)
        self.output_shape_no_batch = self.output_layer.shape[1:]

        super(SeResidualBlock, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.seBlock(x)

        projected_input = self.projectedInput(input_tensor) if \
            K.int_shape(input_tensor)[-1] != self.filter_3 else input_tensor
        shortcut = Add()([projected_input, x])
        shortcut = Activation(activation='relu')(shortcut)

        return shortcut


class SeResnet(tf.keras.Model):
    def __init__(self, input_shape, num_blocks, reduction_ratio=16, **kwargs):
        super(SeResnet, self).__init__(**kwargs)
        self.input_layer = Input(input_shape)   # , batch_size=1
        self.blocks_1, self.blocks_2, self.blocks_3, self.blocks_4 = num_blocks
        self.reduction_ratio = reduction_ratio

        self.conv1, lastOutShape = self._stageBlock(input_shape, 64, 0, stage='1')
        self.conv2, lastOutShape = self._stageBlock(lastOutShape, [64, 64, 256], self.blocks_1, stage='2')
        self.conv3, lastOutShape = self._stageBlock(lastOutShape, [128, 128, 512], self.blocks_2, stage='3')
        self.conv4, lastOutShape = self._stageBlock(lastOutShape, [256, 256, 1024], self.blocks_3, stage='4')
        self.conv5, lastOutShape = self._stageBlock(lastOutShape, [512, 512, 2048], self.blocks_4, stage='5')
        self.output_layer = self.call(self.input_layer)

        super(SeResnet, self).__init__(
            self.input_layer,
            self.output_layer,
            **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

    def _stageBlock(self, input_shape, filter_sizes, blocks, stage=''):
        strides = 2 if stage != '2' else 1
        if stage != '1':
            tmpSB = SeResidualBlock(input_shape, filter_sizes, strides, self.reduction_ratio)
            lastOutShape = tmpSB.output_shape_no_batch
            layers = [tmpSB]

            for i in range(blocks - 1):
                tmpSB = SeResidualBlock(lastOutShape, filter_sizes, reduction_ratio=self.reduction_ratio)
                lastOutShape = tmpSB.output_shape_no_batch
                layers.append(tmpSB)

        else:
            layers = [
                Conv2dBn(input_shape, filter_sizes, (7, 7), strides=strides, padding='same'),
                MaxPooling2D((3, 3), strides=2, padding='same')
            ]
        
        convStage = Sequential(layers, name='conv'+str(stage))
        lastOutShape = convStage.output_shape[1:]

        return convStage, lastOutShape


def se_resnet50():
    return SeResnet((224, 224, 3), [3, 4, 6, 3])

def se_resnet101():
    return SeResnet((224, 224, 3), [3, 4, 23, 3])

def se_resnet152():
    return SeResnet((224, 224, 3), [3, 8, 36, 3])


seResnet50 = se_resnet50()
seResnet50.build()
print(seResnet50.summary())
