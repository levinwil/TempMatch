import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
svhn_mean = (0.4380, 0.4440, 0.4730)
svhn_std = (0.1751, 0.1771, 0.1744)
stl10_mean = [x / 255 for x in [112.4, 109.1, 98.6]]
stl10_std = [x / 255 for x in [68.4, 66.6, 68.5]]


def get_cifar10(args, root):
  transform_labeled = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=32,
                            padding=int(32*0.125),
                            padding_mode='reflect'),
      transforms.ToTensor(),
      transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
  ])
  transform_val = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
  ])
  base_dataset = datasets.CIFAR10(root, train=True, download=True)

  train_labeled_idxs, train_unlabeled_idxs = x_u_split(
      args, base_dataset.targets)

  train_labeled_dataset = CIFAR10SSL(
      root, train_labeled_idxs, train=True,
      transform=transform_labeled)

  train_unlabeled_dataset = CIFAR10SSL(
      root, train_unlabeled_idxs, train=True,
      transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

  test_dataset = datasets.CIFAR10(
      root, train=False, transform=transform_val, download=False)

  return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_svhn(args, root):
  transform_labeled = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
      transforms.ToTensor(),
      transforms.Normalize(mean=svhn_mean, std=svhn_std)
  ])
  transform_val = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean=svhn_mean, std=svhn_std)])
  base_dataset = datasets.SVHN(root, split="train", download=True)

  train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.labels)

  train_labeled_dataset = SVHNSSL(root,
                                  train_labeled_idxs,
                                  split="train",
                                  transform=transform_labeled)

  train_unlabeled_dataset = SVHNSSL(root,
                                    train_unlabeled_idxs,
                                    split="train",
                                    transform=TransformFixMatch(mean=svhn_mean, std=svhn_std))

  test_dataset = datasets.SVHN(root, split="test", transform=transform_val, download=True)

  return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_stl10(args, root):
  transform_labeled = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
      transforms.ToTensor(),
      transforms.Normalize(mean=stl10_mean, std=stl10_std)
  ])
  transform_val = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean=stl10_mean, std=stl10_std)])
  base_dataset = datasets.STL10(root, split="train", download=True)

  train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.labels)

  train_labeled_dataset = STL10SSL(root,
                                  train_labeled_idxs,
                                  split="train",
                                  transform=transform_labeled)

  train_unlabeled_dataset = STL10SSL(root,
                                     train_unlabeled_idxs,
                                     split="train",
                                     unlabeled=True,
                                     transform=TransformFixMatch(mean=stl10_mean, std=stl10_std))

  test_dataset = datasets.STL10(root, split="test", transform=transform_val, download=True)

  return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

  transform_labeled = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=32,
                            padding=int(32*0.125),
                            padding_mode='reflect'),
      transforms.ToTensor(),
      transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

  transform_val = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

  base_dataset = datasets.CIFAR100(
      root, train=True, download=True)

  train_labeled_idxs, train_unlabeled_idxs = x_u_split(
      args, base_dataset.targets)

  train_labeled_dataset = CIFAR100SSL(
      root, train_labeled_idxs, train=True,
      transform=transform_labeled)

  train_unlabeled_dataset = CIFAR100SSL(
      root, train_unlabeled_idxs, train=True,
      transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

  test_dataset = datasets.CIFAR100(
      root, train=False, transform=transform_val, download=False)

  return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
  label_per_class = args.num_labeled // args.num_classes
  labels = np.array(labels)
  labeled_idx = []
  # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
  unlabeled_idx = np.array(range(len(labels)))
  for i in range(args.num_classes):
    idx = np.where(labels == i)[0]
    idx = np.random.choice(idx, label_per_class, False)
    labeled_idx.extend(idx)
  labeled_idx = np.array(labeled_idx)
  assert len(labeled_idx) == args.num_labeled

  if args.expand_labels or args.num_labeled < args.batch_size:
    num_expand_x = math.ceil(
        args.batch_size * args.eval_step / args.num_labeled)
    labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
  np.random.shuffle(labeled_idx)
  return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
  def __init__(self, mean, std):
    self.weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect')])
    self.strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10)])
    self.normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

  def __call__(self, x):
    weak = self.weak(x)
    strong = self.strong(x)
    return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
  def __init__(self, root, indexs, train=True,
               transform=None, target_transform=None,
               download=False):
    super().__init__(root, train=train,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    if indexs is not None:
      self.data = self.data[indexs]
      self.targets = np.array(self.targets)[indexs]

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class CIFAR100SSL(datasets.CIFAR100):
  def __init__(self, root, indexs, train=True,
               transform=None, target_transform=None,
               download=False):
    super().__init__(root, train=train,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    if indexs is not None:
      self.data = self.data[indexs]
      self.targets = np.array(self.targets)[indexs]

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class SVHNSSL(datasets.SVHN):

  def __init__(self,
               root,
               indexs,
               split="train",
               transform=None,
               target_transform=None,
               download=False):
    super().__init__(root,
                     split=split,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    if indexs is not None:
      self.data = self.data[indexs]
      self.labels = np.array(self.labels)[indexs]
    self.data = self.data.transpose([0, 2, 3, 1])

  def __getitem__(self, index):
    img, target = self.data[index], self.labels[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class STL10SSL(datasets.STL10):

  def __init__(self,
               root,
               indexs,
               split="train",
               unlabeled=False
               transform=None,
               target_transform=None,
               download=False):
    super().__init__(root,
                     split=split,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    if indexs is not None:
      self.data = self.data[indexs]
      self.labels = np.array(self.labels)[indexs]
    self.labels = self.labels.astype(np.int64)
    if unlabeled:
      unlabeled_dataset = super().__init__(root, split="unlabeled", download=True)
      self.data = np.concatenate([unlabeled_dataset.data, self.data])
      self.labels = np.concatenate([[None for _ in range(len(unlabeled_dataset.data))], self.labels])
    self.data = self.data.transpose([0, 2, 3, 1])

  def __getitem__(self, index):
    img, target = self.data[index], self.labels[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class PseudoSSL(Dataset):

  def __init__(self,
               data,
               targets):
    self.data = data
    self.targets = targets

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    img, target = self.data[index], np.array(self.targets)[index]

    return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'svhn': get_svhn,
                   'stl-10': get_stl_10,
                   'pseudossl': PseudoSSL}
