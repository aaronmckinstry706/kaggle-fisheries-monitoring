import lasagne
import lasagne.init as init
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import theano.tensor as tensor

def fully_convolutional_smallest(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=16,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=32,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                            W=init.HeNormal())
    hidden_layer_three = layers.batch_norm(hidden_layer_three)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer_three,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

def fully_convolutional_small(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=16,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=32,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer_three = layers.batch_norm(hidden_layer_three)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer_three,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

def fully_convolutional_medium(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=16,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=32,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                            num_filters=64,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer_three = layers.batch_norm(hidden_layer_three)
    
    hidden_layer_four = layers.Conv2DLayer(incoming=hidden_layer_three,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer_four = layers.batch_norm(hidden_layer_four)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer_four,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

def fully_convolutional_big(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=48,
                                          filter_size=(3,3),
                                          stride=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=128,
                                          filter_size=(3,3),
                                          stride=(2,2),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                            num_filters=256,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer_three = layers.batch_norm(hidden_layer_three)
    
    hidden_layer_four = layers.Conv2DLayer(incoming=hidden_layer_three,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer_four = layers.batch_norm(hidden_layer_four)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer_four,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

def fully_convolutional_deep(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=64,
                                          filter_size=(7,7),
                                          stride=(4,4),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=64,
                                          filter_size=(3,3),
                                          stride=(2,2),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer = hidden_layer_two
    num_small_hidden_layers = 5
    for i in range(0, num_small_hidden_layers):
        hidden_layer = layers.Conv2DLayer(incoming=hidden_layer,
                                          num_filters=64,
                                          filter_size=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
        hidden_layer = layers.batch_norm(hidden_layer)
    
    hidden_layer = layers.Conv2DLayer(incoming=hidden_layer,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer = layers.batch_norm(hidden_layer)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

def fully_convolutional_deeper(inputs, image_shape, num_outputs):
    """Builds a fully-convolutional neural network. 
    
    Args:
        image_shape -- A tuple representing the dimensions of a single image.
                       If multiple color channels, then the channel dimension
                       must be first. E.g., (512, 512) or (3, 512, 512)--but
                       not (512, 512, 3) because it would be interpreted as a
                       512-channel image of size 512x3. 
        inputs -- A theano symbolic tensor with one more dimension than
                  image_shape. For example, if image_shape == (224, 224), then
                  'inputs' should be of type theano.tensor.Tensor3. 
    
    Returns:
        network -- An object of a class descended from lasagne.layers.Layer.
                   This object represents the final layer of the network. It has
                   exactly num_classes output neurons. 
    """
    input_layer = layers.InputLayer(shape=(None,) + image_shape,
                                    input_var=inputs)
    
    hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                          num_filters=64,
                                          filter_size=(7,7),
                                          stride=(4,4),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_one = layers.batch_norm(hidden_layer_one)
    
    hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                          num_filters=64,
                                          filter_size=(3,3),
                                          stride=(2,2),
                                          pad='same',
                                          W=init.HeNormal())
    hidden_layer_two = layers.batch_norm(hidden_layer_two)
    
    hidden_layer = hidden_layer_two
    num_small_hidden_layers = 10
    for i in range(0, num_small_hidden_layers):
        hidden_layer = layers.Conv2DLayer(incoming=hidden_layer,
                                          num_filters=64,
                                          filter_size=(3,3),
                                          pad='same',
                                          W=init.HeNormal())
        hidden_layer = layers.batch_norm(hidden_layer)
    
    hidden_layer = layers.Conv2DLayer(incoming=hidden_layer,
                                            num_filters=num_outputs,
                                            filter_size=(3,3),
                                            pad='same',
                                          W=init.HeNormal())
    hidden_layer = layers.batch_norm(hidden_layer)
    
    output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer,
                                                      pool_function=tensor.max)
    
    output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                            nonlinearity=nonlinearities.softmax)
    
    return output_layer

