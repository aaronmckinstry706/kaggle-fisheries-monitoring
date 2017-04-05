import keras
import math
import numpy
import random
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

def get_training_iterator(directory, desired_width, batch_size):
    training_generator = image.ImageDataGenerator(
        rotation_range=15.0,
        zoom_range=[math.sqrt(2)*math.cos(math.pi/4 - 15.0/180.0*math.pi),
                    math.sqrt(2)*math.cos(math.pi/4 - 15.0/180.0*math.pi)],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing=resize_and_random_crop(desired_width))
    return training_generator.flow_from_directory(
        directory, batch_size=batch_size,
        target_size=(desired_width, desired_width), 
        shuffle=True)

def threaded_generator(image_directory, num_threads=8, **kwargs):
    import Queue
    queue = Queue.Queue(maxsize=50)
    sentinels = [object() for i in range(0,num_threads)]

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
