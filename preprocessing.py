import math
import numpy
import Queue
import random
import threading
import time

import keras.preprocessing.image as image
import scipy.misc as misc
import scipy.ndimage as ndimage

def resize_and_random_crop(desired_width):
    def function(image):
        if image.shape[0] <= image.shape[1]:
            min_dim = 0
        else:
            min_dim = 1
        
        scale_factor = float(desired_width)/image.shape[min_dim]
        new_dimensions = [
            int(round(image.shape[0]*scale_factor)),
            int(round(image.shape[1]*scale_factor)),
            3]
        new_dimensions[min_dim] = desired_width
        image = misc.imresize(image, size=tuple(new_dimensions))
        
        max_length = max(new_dimensions[0], new_dimensions[1])
        start_index = random.choice(range(0, max_length - desired_width + 1))
        
        if min_dim == 0:
            return image[:,start_index:start_index+desired_width,:]
        else:
            return image[start_index:start_index+desired_width,:,:]
    
    return function

def resize_and_center_crop(desired_width):
    def function(image):
        if image.shape[0] <= image.shape[1]:
            min_dim = 0
        else:
            min_dim = 1
        
        scale_factor = float(desired_width)/image.shape[min_dim]
        new_dimensions = [
            int(round(image.shape[0]*scale_factor)),
            int(round(image.shape[1]*scale_factor)),
            3]
        new_dimensions[min_dim] = desired_width
        image = misc.imresize(image, size=tuple(new_dimensions))
        
        max_length = max(new_dimensions[0], new_dimensions[1])
        start_index = int(math.floor((max_length - 512)/2.0))
        
        if min_dim == 0:
            return image[:,start_index:start_index+desired_width,:]
        else:
            return image[start_index:start_index+desired_width,:,:]
    
    return function

def get_training_generator(directory, desired_width, batch_size):
    training_generator = image.ImageDataGenerator(
        rotation_range=15.0,
        zoom_range=[1.0, math.sqrt(2)*math.cos(math.pi/4 - 15.0/180.0*math.pi)],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=resize_and_random_crop(desired_width))
    return training_generator.flow_from_directory(
        directory, batch_size=batch_size,
        target_size=(desired_width, desired_width), 
        shuffle=True)

def get_validation_generator(directory, desired_width, batch_size):
    validation_generator = image.ImageDataGenerator(
        preprocessing_function=resize_and_center_crop(desired_width))
    return validation_generator.flow_from_directory(
        directory, batch_size=batch_size,
        target_size=(desired_width, desired_width), shuffle=False)

def get_test_generator(directory, desired_width, batch_size):
    test_generator = image.ImageDataGenerator(
        preprocessing_function=resize_and_center_crop(desired_width))
    return test_generator.flow_from_directory(
        directory, batch_size=batch_size, class_mode=None,
        target_size=(desired_width, desired_width), shuffle=False)

def get_threaded_generator(directory, desired_width, batch_size, type,
                          num_threads=8):
    generator_getters = {'training': get_training_generator,
                        'validation': get_validation_generator,
                        'test': get_test_generator}
    generator_getter = generator_getters[type]
    
    data_generator = generator_getter(directory, desired_width,
                                    batch_size)
    num_data_points = len(data_generator.filenames)
    
    queue = Queue.Queue(maxsize=50)
    sentinel = object()
    stop_event = threading.Event()
    
    # define producer (putting items into queue)
    def producer():
        while not stop_event.is_set():
            item = next(data_generator)
            success = False
            while not success and not stop_event.is_set():
                try:
                    queue.put_nowait(item)
                    success = True
                except Queue.Full:
                    time.sleep(0.05)

    # start producers (in background threads)
    threads = [threading.Thread(target=producer)
               for i in range(0, num_threads)]
    for thread in threads:
        thread.daemon = True
        thread.start()
    
    # run as consumer (read items from queue, in current thread)
    count = 0
    item = queue.get()
    while item != sentinel:
        yield item
        count += item[0].shape[0]
        if count >= num_data_points:
            stop_event.set()
            break
        queue.task_done()
        item = queue.get()
