import os
import csv
import argparse

import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            new_row = row
            for i in range(3):
                dirs = os.path.normpath(new_row[i]).split(os.sep)
                if len(dirs) > 4:
                    new_row[i] = os.path.join(*dirs[-4:])
            new_rows.append(new_row)

    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

def get_samples(datasets, split=0.2, base_url='data'):
    '''
    returns training and validation samples
    '''
    samples = []
    prev_count = 0
    for dataset in datasets:
        url = get_csv(dataset, base_url=base_url)
        with open(url) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        print('Read {} samples from {}'.format(len(samples)-prev_count, url))
        prev_count = len(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=split)
    return train_samples, validation_samples

def generator(samples, batch_size=32):
    '''
    generator for dataset
    '''
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                '''
                format: center image, left image, right image, angle, throttle, break, speed
                '''
                center_image = mpimg.imread(batch_sample[0])
                angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset processor')
    parser.add_argument("--fix_paths", help="converts abs to relative paths",
                        action="store_true")
    parser.add_argument("--display", help="displays images",
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
    else:
        print("I did nothing")
