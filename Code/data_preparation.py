import ee
import numpy as np
from PIL import Image
import requests
import math
from tqdm import tqdm

'''
Clip an Earth Engine Image to an area of interest
'''
def clip_image(image, aoi):
    return ee.Image(image).clip(aoi)

'''
Create a RGB composite from the diferent bands 
For Sentinel-1 data, the standard way to make the RGB composite is VV for red, VH for green and VV/VH for blue
'''
def to_rgb(image):
    return ee.Image.rgb(image.select('VV'),                            
                        image.select('VH'),                            
                        image.select('VV').divide(image.select('VH')))

'''
Convert an Earth Engine Image into a Pillow Image
'''
def to_pillow(image):
    url = image.getThumbURL({'min': [-20, -20, 0], 'max': [0, 0, 2]})
    return Image.open(requests.get(url, stream=True).raw)

'''
Crop the center of a Pillow Image 
'''
def crop_image(image, new_width, new_height):
    image_width, image_height = image.size
    left = round((image_width - new_width)/2)
    top = round((image_height - new_height)/2)
    x_right = round(image_width - new_width) - left
    x_bottom = round(image_height - new_height) - top
    right = image_width - x_right
    bottom = image_height - x_bottom
    return image.crop((left, top, right, bottom))

'''
Convert a Pillow Image into a Numpy Array
'''
def to_array(image):
    data = np.array(image)
    return data[:, :, 0]

'''
Convert a Numpy Array into a Pilow Image
'''
def to_image(array):
    image = Image.fromarray(array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

'''
Convert Earth Engine Images into Numpy Arrays
'''
def convert_data(images):
    nr_images = images.length().getInfo()
    data = []
    for i in tqdm(range(nr_images)):
        image = ee.Image(images.get(i))
        image_rgb = to_rgb(image)
        image_pillow = to_pillow(image_rgb)
        image_cropped = crop_image(image_pillow, 1024, 1024)
        image_data = to_array(image_cropped)
        data.append(image_data)
    data = np.array(data)
    return data.astype(np.int16)

'''
Separate images into pairs to detect changes
'''
def to_pairs(data):
    new_array = []
    for i in range(data.shape[0] - 1):
        pair = [data[i], data[i + 1]]
        new_array.append(pair)
        pairs = np.array(new_array)
    return pairs

'''
Create a set of changes to make the problem supervised
'''
def to_supervised(data, threshold):
    changes = []
    for pair in data:
        ratio = np.subtract(pair[0], pair[1])
        abs = np.absolute(ratio)
        abs[abs < threshold] = 0  # set pixels with value < threshold to 0 
        abs[abs >= threshold] = 1 # set pixels with value >= threshold to 1 
        changes.append(abs)
    return np.array(changes)

'''
Slice an array into a smaller array
'''
def slice_array(array, width, height):
    slices = []
    for x in range(0, array.shape[0], height):
        for y in range(0, array.shape[1], width):
            slices.append(array[x: x + height, y: y + width])
    return np.array(slices)

'''
Slice images and group the data into the shape (number of pairs, number of slices per pair, number of arrays per pair, slice width, slice height)
'''
def slice_data(data, image_size):
    slices = []
    for pair in data:
        slices1 = slice_array(pair[0], image_size, image_size) # slices of the first array of the pair
        slices2 = slice_array(pair[1], image_size, image_size) # slices of the second array of the pair
        grouped_slices = [] # list with the slices of the pair grouped
        for s in range(len(slices1)):
            grouped_slices.append([slices1[s], slices2[s]])
        slices.append(grouped_slices)
    return np.array(slices)

'''
Slice pairs changes into the shape (number of changes, number of slices, slice width, slice height)
'''
def slice_changes(changes, image_size=128):
    slices = []
    for array in changes:
        slices.append(slice_array(array, image_size, image_size))
    return np.array(slices)

'''
Separate each pair of images
'''
def separate_pairs(data, changes):
    data = data.reshape(-1, *data.shape[-3:])
    changes = changes.reshape(-1, *changes.shape[-2:])
    return data, changes

'''
Normalize pixel values
The images are in gray scale, so the values are between [0, 255]
''' 
def normalize_data(data):
    return data.astype('float32') / 255

'''
Denormalize pixel values
'''
def denormalize_data(data):
    return data.astype('float32') * 255

'''
Reshape data to fit the Deep Learning model
'''
def reshape_data(data):
    new_data = []
    for pair in data:
        new_data.append(np.stack((pair[0], pair[1]), axis=-1))
    return np.array(new_data)

'''
Shuffle the data from the pairs of images and changes in unison 
'''
def shuffle_data(data, changes):
    seed = np.random.randint(0, 100000) 
    np.random.seed(seed)  
    np.random.shuffle(data)  
    np.random.seed(seed)  
    np.random.shuffle(changes)
    return data, changes  

'''
Split data into training and testing sets
'''
def split_data(data, changes, percentage):
    nr_images, _, _, _ = data.shape
    train_size = math.floor(nr_images * percentage)
    x_train, x_test = data[:train_size,:], data[train_size:,:]
    y_train, y_test = changes[:train_size,:], changes[train_size:,:]
    return x_train, y_train, x_test, y_test