import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from data_reader import data_reader

TRAINING_IMAGES_PATH = Path("training")


class ImageDataset(Dataset):

    def __init__(self, start_sequence: int, end_sequence: int):
        """
        Initializes the image arrays from the path to the training images dataset by setting their height and width
        to 100 px which is the same shape as the test set.
        :param start_sequence: starting folder position where we choose our set of images to train from the dataset
        :param end_sequence: ending folder position where we choose our set of images to train from the dataset
        """
        self.image_arrays = []

        im_shape = 100
        resize_transforms = transforms.Compose(
            [transforms.Resize(size=im_shape),
             transforms.CenterCrop(size=(im_shape, im_shape)),
             ])

        for folder_number in range(start_sequence, end_sequence):
            folder_number = '{:0>3}'.format(folder_number)
            folder_name = os.path.join(TRAINING_IMAGES_PATH, folder_number)

            for image_file_name in os.listdir(folder_name):
                with Image.open(Path(folder_name, image_file_name)) as im:
                    im = resize_transforms(im)
                    img = np.asarray(im, dtype=np.uint8)
                    self.image_arrays.append(img)

    def __len__(self) -> int:
        """
        :return: length of training images
        """
        return self.image_arrays.__len__()

    def __getitem__(self, index: int) -> tuple:
        """
        processes an image to extract the input, known and target arrays

        :param index: index of the image in the array self.image_arrays (ALL training images)
        :return: the input array, known array and target array as well as the index of the image as a tuple
        """
        image_array = self.image_arrays[index]

        # Random offset like test set from 0 to 8 like Test set
        random_offset = np.random.randint(0, 9), np.random.randint(0, 9)

        # Random offset like test set from 2 to 6 like Test set
        random_spacing = np.random.randint(2, 7), np.random.randint(2, 7)

        input_array, known_array, target_array = data_reader(image_array, random_offset, random_spacing)

        return input_array, known_array, target_array, index


def collate_fn(batch_as_list: list):
    """
    stacks the input, known and target arrays as well as the labels into single tensors and returns a tuple
    :param batch_as_list:
    :return: tuple of stacked input, known and target arrays and also the stacked labels
    """
    max_x = 100
    max_y = 100
    n_feature_channels = 3
    n_samples = len(batch_as_list)

    input_arrays = [sample[0] for sample in batch_as_list]
    known_arrays = [sample[1] for sample in batch_as_list]
    target_arrays = [sample[2] for sample in batch_as_list]
    labels = [sample[3] for sample in batch_as_list]

    stacked_input_sequences = torch.zeros(size=(n_samples, n_feature_channels, max_y, max_x), dtype=torch.float32)

    for i, input_array in enumerate(input_arrays):
        stacked_input_sequences[i, :n_feature_channels, :max_x, :max_y] = torch.tensor(input_array)

    stacked_known_sequences = torch.zeros(size=(n_samples, n_feature_channels, max_y, max_x), dtype=torch.float32)

    for i, known_array in enumerate(known_arrays):
        stacked_known_sequences[i, :n_feature_channels, :max_x, :max_y] = torch.tensor(known_array)

    stacked_target_sequences = torch.zeros(
        size=(n_samples, np.max([target_arr.shape[0] for target_arr in target_arrays])))

    for i, target_array in enumerate(target_arrays):
        stacked_target_sequences[i, :target_array.shape[0]] = torch.tensor(target_array)

    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)

    return stacked_input_sequences, stacked_known_sequences, stacked_target_sequences, stacked_labels


class TestDataset(Dataset):
    def __init__(self, test_set):
        self.input_arrays = []
        self.known_arrays = []
        self.offsets = []
        self.spacings = []
        self.sample_ids = []

        for sample_number in range(8663):  # 8663 is our test set length of samples
            input_array = test_set["input_arrays"][sample_number]
            known_array = test_set['known_arrays'][sample_number]
            offset = test_set['offsets'][sample_number]
            spacing = test_set['spacings'][sample_number]
            sample_id = test_set['sample_ids'][sample_number]

            self.input_arrays.append(input_array)
            self.known_arrays.append(known_array)
            self.offsets.append(offset)
            self.spacings.append(spacing)
            self.sample_ids.append(sample_id)

    def __len__(self):
        return self.input_arrays.__len__()

    def __getitem__(self, index):
        input_array = self.input_arrays[index]
        known_array = self.known_arrays[index]
        offset = self.offsets[index]
        spacing = self.spacings[index]
        sample_id = self.sample_ids[index]

        # input_array, known_array, target_array = data_reader(input_array, offset, spacing)

        return input_array, known_array, sample_id


def collate_fn_test(batch_as_list: list):
    """
    stacks the input, known and target arrays as well as the labels into single tensors and returns a tuple
    :param batch_as_list:
    :return: tuple of stacked input, known and target arrays and also the stacked labels
    """
    max_x = 100
    max_y = 100
    n_feature_channels = 3
    n_samples = len(batch_as_list)

    input_arrays = [sample[0] for sample in batch_as_list]
    known_arrays = [sample[1] for sample in batch_as_list]
    labels = [sample[2] for sample in batch_as_list]

    stacked_input_sequences = torch.zeros(size=(n_samples, n_feature_channels, max_y, max_x), dtype=torch.float32)

    for i, input_array in enumerate(input_arrays):
        stacked_input_sequences[i, :n_feature_channels, :max_x, :max_y] = torch.tensor(input_array)

    stacked_known_sequences = torch.zeros(size=(n_samples, n_feature_channels, max_y, max_x), dtype=torch.float32)

    for i, known_array in enumerate(known_arrays):
        stacked_known_sequences[i, :n_feature_channels, :max_x, :max_y] = torch.tensor(known_array)

    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)

    return stacked_input_sequences, stacked_known_sequences, stacked_labels
# if __name__ == "__main__":
#     # data = ImageDataset(2, 4)
#     # print(data.__len__())
#
#     list = []
#
#     for i in range(3):
#         list.append((torch.rand(size=(3, 100, 100)), torch.rand(size=(3, 100, 100)), torch.rand(50)))
#
#     print(collate_fn(list)[2].shape)
