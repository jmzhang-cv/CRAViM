import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dirs, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirs) or os.path.islink(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnalignedDataset(data.Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.chanels = opt.input_nc
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
                and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # Resize and pad as needed
        if A_img.size != self.opt.image_size:
            if A_img.size[0] < self.opt.image_size or A_img.size[1] < self.opt.image_size:
                padding = transforms.Pad((
                    (self.opt.image_size - A_img.size[0]) // 2,
                    (self.opt.image_size - A_img.size[1]) // 2,
                ), fill=0)
                A_img = padding(A_img)
                B_img = padding(B_img)
            else:
                resize = transforms.Resize(self.opt.image_size)
                A_img = resize(A_img)
                B_img = resize(B_img)

        if self.opt.phase == "train":
            # p1 = random.randint(0, 1)
            # transform = transforms.Compose([
            #     transforms.RandomHorizontalFlip(p1),
            #     transforms.Pad((20, 20, 20, 20), fill=0),
            #     transforms.RandomCrop(self.opt.image_size),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])
            # seed = np.random.randint(999999999)
            # torch.random.manual_seed(seed)  # apply this seed to img tranfsorms
            # A = transform(A_img)
            # if self.opt.pair:
            #     torch.random.manual_seed(seed)  # apply this seed to img tranfsorms
            # B = transform(B_img)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            A = transform(A_img)
            B = transform(B_img)

            if self.chanels == 1:
                A = A[0]
                A = A.reshape([-1, A.shape[0], A.shape[1]])
                B = B[0]
                B = B.reshape([-1, B.shape[0], B.shape[1]])
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            A = transform(A_img)
            B = transform(B_img)
            if self.chanels == 1:
                A = A[0]
                A = A.reshape([-1, A.shape[0], A.shape[1]])
                B = B[0]
                B = B.reshape([-1, B.shape[0], B.shape[1]])
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
