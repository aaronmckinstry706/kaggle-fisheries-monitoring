import typing

def read_config_file(filename):
    # type: (str) -> typing.Dict[str, typing.Any]
    config_params = {} # type: typing.Dict[str, typing.Any]
    
    identity_function = lambda x: x
    processing_functions = {
        'training_directory': identity_function,
        'validation_directory': identity_function,
        'test_directory': identity_function,
        'num_threads_for_preprocessing': int,
        'image_width': int,
        'initial_learning_rate': float,
        'momentum': float,
        'batch_size': int,
        'weight_decay': float,
        'num_classes' : int,
        'num_epochs' : int} # type: typing.Dict[str, typing.Callable]
    for line in open(filename, 'r'):
        if '#' in line:
            line = line.split('#')[0]
        if '=' in line:
            [name, value] = line.split('=')
            name = name.strip()
            value = value.strip()
            if processing_functions.has_key(name):
                config_params[name] = processing_functions[name](value)
    return config_params

