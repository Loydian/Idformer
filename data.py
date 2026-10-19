import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import Sampler
import random
import os

RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS


def read_imgs(train_image, fix_image_width, fix_image_height, pid2label):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), RESAMPLE_LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)


class SYSUData(data.Dataset):
    def __init__(self, data_dir, img_size=(144, 288),
                 transform_color1=None, transform_color2=None, transform_thermal=None):
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']

        # load id info
        file_path_train = os.path.join(data_dir, 'exp', 'train_id.txt')
        file_path_val = os.path.join(data_dir, 'exp', 'val_id.txt')
        with open(file_path_train, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_train = ["%04d" % x for x in ids]

        with open(file_path_val, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_val = ["%04d" % x for x in ids]

        # combine train and val split
        id_train.extend(id_val)

        files_rgb = []
        rgb_cams = []
        files_ir = []
        ir_cams = []
        for id in sorted(id_train):
            for cam in rgb_cameras:
                img_dir = os.path.join(data_dir, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                    files_rgb.extend(new_files)
                    rgb_cams.extend([int(cam[-1]) - 1] * len(new_files))

            for cam in ir_cameras:
                img_dir = os.path.join(data_dir, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)
                    ir_cams.extend([int(cam[-1]) - 1] * len(new_files))

        # relabel
        pid_container = set()
        for img_path in files_ir:
            pid = int(img_path[-13:-9])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        self.train_color_image, self.train_color_label = read_imgs(files_rgb, img_size[0], img_size[1], pid2label)
        self.train_thermal_image, self.train_thermal_label = read_imgs(files_ir, img_size[0], img_size[1], pid2label)
        self.rgb_cams = np.array(rgb_cams)
        self.ir_cams = np.array(ir_cams)

        self.transform_color1 = transform_color1
        self.transform_color2 = transform_color2
        self.transform_thermal = transform_thermal
        self.cIndex = None
        self.tIndex = None

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_color_image[self.cIndex[index]], \
                              self.train_color_label[self.cIndex[index]], \
                              self.rgb_cams[self.cIndex[index]]
        img2, target2, cam2 = self.train_thermal_image[self.tIndex[index]], \
                              self.train_thermal_label[self.tIndex[index]], \
                              self.ir_cams[self.tIndex[index]]

        img1_0 = self.transform_color1(img1)
        img1_1 = self.transform_color2(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, img_size=(144, 288),
                 transform_color1=None, transform_color2=None, transform_thermal=None):
        # Load training images (path) and labels
        train_color_list = os.path.join(data_dir, 'idx', 'train_visible_{}'.format(trial) + '.txt')
        train_thermal_list = os.path.join(data_dir, 'idx', 'train_thermal_{}'.format(trial) + '.txt')

        color_img_file, self.train_color_label = RegDBData.load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = RegDBData.load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(os.path.join(data_dir, color_img_file[i]))
            img = img.resize(img_size, RESAMPLE_LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        self.train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(os.path.join(data_dir, thermal_img_file[i]))
            img = img.resize(img_size, RESAMPLE_LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        self.train_thermal_image = np.array(train_thermal_image)

        self.transform_color1 = transform_color1
        self.transform_color2 = transform_color2
        self.transform_thermal = transform_thermal
        self.cIndex = None
        self.tIndex = None

    @staticmethod
    def load_data(input_data_path):
        data_file_list = open(input_data_path, 'r').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

        return file_image, file_label

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform_color1(img1)
        img1_1 = self.transform_color2(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2, -1, -1

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, test_cam=None, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), RESAMPLE_LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.test_cam = test_cam
        self.transform = transform

    def __getitem__(self, index):
        if self.test_cam is not None:
            cam = self.test_cam[index]
        else:
            cam = -1
        img, target = self.test_image[index], self.test_label[index]
        img = self.transform(img)
        return img, target, cam

    def __len__(self):
        return len(self.test_image)


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


def process_query_sysu(data_path):
    ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp', 'test_id.txt')
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]) - 1, int(img_path[-13:-9])
        query_id.append(pid)
        query_cam.append(camid)
    return files_ir, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all'):
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
    else:
        raise Exception(f"Unexpected mode: {mode}")

    file_path = os.path.join(data_path, 'exp', 'test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))

    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]) - 1, int(img_path[-13:-9])
        gall_id.append(pid)
        gall_cam.append(camid)
    return files_rgb, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = os.path.join(img_dir, 'idx', 'test_visible_{}'.format(trial) + '.txt')
    elif modal == 'thermal':
        input_data_path = os.path.join(img_dir, 'idx', 'test_thermal_{}'.format(trial) + '.txt')
    else:
        raise Exception(f"Invaild modal: {modal}")

    data_file_list = open(input_data_path, 'r').read().splitlines()
    # Get full list of image and labels
    file_image = [os.path.join(img_dir, s.split(' ')[0]) for s in data_file_list]
    file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)


class IdentitySampler(Sampler):
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
