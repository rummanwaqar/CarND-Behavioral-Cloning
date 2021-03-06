import os
import csv
import argparse

import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

CSV_FILE_NAME = 'driving_log.csv'

def get_dataset_names(base_url='data'):
    '''
    gets list of datasets in base_url
    '''
    return [x for x in os.listdir(base_url)
            if os.path.exists(os.path.join(base_url, x, CSV_FILE_NAME))]

def get_csv(dataset, base_url='data'):
    '''
    returns csv url
    assumes path is base_url/dataset/driving_log.csv
    '''
    url = os.path.join(base_url, dataset, CSV_FILE_NAME)
    if os.path.exists(url):
        return url
    return None

def fix_csv_paths(csv_file):
    '''
    Converts urls from abs to relative format (data/dataset/IMG/*.jpg)
    '''
    new_rows = []
    base_url = csv_file.split(os.sep)[:2]
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            new_row = row
            for i in range(3):
                dirs = os.path.normpath(new_row[i]).strip().split(os.sep)
                if len(dirs) > 4:
                    new_row[i] = os.path.join(*dirs[-4:])
                elif len(dirs) < 3:
                    new_row[i] = os.path.join(*base_url, *dirs)
            new_rows.append(new_row)

    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

def get_samples(datasets, split=0.2, base_url='data', all=True, correction=0.25, balanced=True):
    '''
    returns training and validation samples
    csv format: center image, left image, right image, angle, throttle, break, speed
    output sample format: image url, angle
    '''
    samples = []
    for dataset in datasets:
        count = 0
        url = get_csv(dataset, base_url=base_url)
        with open(url) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                angle = float(line[3])
                count += 1
                samples.append([line[0], angle])
                if all:
                    count += 2
                    samples.append([line[1], angle+correction])
                    samples.append([line[2], angle-correction])
        print('Read {} samples from {}'.format(count, url))
    if balanced:
        samples = balance_samples(samples, [0, correction, -correction])
    train_samples, validation_samples = train_test_split(samples, test_size=split)
    return train_samples, validation_samples

def balance_samples(samples, bins, bin_width=0.03, max=2000):
    '''
    applies a max limit on samples within bin_width of bins
    '''
    samples = sklearn.utils.shuffle(samples)
    balanced = []
    bins = np.array(bins)
    bin_counts = np.zeros_like(bins)
    for sample in samples:
        in_bin = np.absolute(bins - sample[1]) <= bin_width
        if np.sum(in_bin) > 0:
            idx = np.argmax(in_bin)
            if bin_counts[idx] < max:
                balanced.append(sample)
                bin_counts[idx] += 1
        else:
            balanced.append(sample)
    return balanced

def preprocess_image(img, width=64, height=64, crop=(65,20)):
    '''
    preprocesses image
    RGB -> YUV
    Crop away top and bottom regions
    Rescale to 66, 200
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = img[crop[0]:-crop[1], :]
    img = cv2.resize(img, (width, height))
    return img

def flip_image(image, angle, prob=0.3):
    '''
    flips the image and steering angle left to right
    '''
    if np.random.rand() < prob:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle

def translate_image(image, angle, x_range, angle_factor=0.0025):
    '''
    translates the image within x_range
    modifies angle using angle_factor
    '''
    translation = x_range*(np.random.rand()-0.5)
    M = np.float32([[1,0,translation], [0,1,0]])
    angle += translation * angle_factor
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image, angle

def augment(img, angle):
    '''
    augment the image
    * flip
    * x-translate
    '''
    image, angle = flip_image(img, angle, prob=0.3)
    image, angle = translate_image(img, angle, x_range=100)
    return image, angle

def generator(samples, batch_size=32, is_training=True, augment_prob=0.5):
    '''
    generator for dataset
    '''
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:
                '''
                format: image, angle
                '''
                image = mpimg.imread(batch_sample[0])
                angle = float(batch_sample[1])
                if is_training and np.random.rand() > augment_prob:
                    image, angle = augment(image, angle)
                images.append(preprocess_image(image))
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def distribution(balanced=True):
    '''
    displays image distribution
    '''
    fig = plt.figure()
    samples, _ = get_samples(datasets=get_dataset_names(),
                               split=0.0,
                               base_url='data',
                               balanced=balanced)
    angles = np.array([float(x[1]) for x in samples])
    plt.hist(angles, 50, rwidth=0.5)
    plt.title('Dataset Distribution')
    plt.xlabel('Driving angle')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset processor')
    parser.add_argument("--fix_paths", help="converts abs to relative paths",
                        action="store_true")
    parser.add_argument("--display", help="displays images",
                        action="store_true")
    parser.add_argument("--dist", help="shows dataset distribution",
                        action="store_true")
    args = parser.parse_args()

    if args.fix_paths:
        print("Converting all abs paths to relative.")
        for dataset in get_dataset_names():
            url = get_csv(dataset)
            print("\tProcessing", url)
            fix_csv_paths(url)
    elif args.display:
        training_samples, validation_samples = get_samples(datasets=get_dataset_names(),
                                                           split=0.2,
                                                           base_url='data')

        training_generator = generator(training_samples, batch_size=8)

        X, y = next(training_generator)

        fig = plt.figure(figsize=(9,9))

        for i in range(8):
            plt.subplot(3,3,i+1)
            plt.title("{}: {:.3f}".format(i+1, y[i]))
            plt.imshow(X[i])
        plt.show()
    elif args.dist:
        distribution()
    else:
        print("I did nothing")
