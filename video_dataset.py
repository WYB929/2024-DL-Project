import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import re
import os
from torchvision import transforms

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the video folders.
            mode (string): 'train', 'val', 'unlabeled', and 'hidden' to specify the dataset to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.extract_video_number = lambda path: int(re.search(r'video_(\d+)', path).group(1))
        self.video_folders = sorted(
            [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))],
            key=self.extract_video_number
        )
        # self.video_folders = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.mode = mode
        self.extract_image_number = lambda path: int(re.search(r'image_(\d+)\.png', path).group(1))

    def __len__(self):
        return len(self.video_folders)
    
    def video_folders(self):
        return self.video_folders

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frames = sorted([os.path.join(video_folder, img) for img in os.listdir(video_folder) if img.endswith('.png')], key=self.extract_image_number)

        images = [Image.open(frame).convert('RGB') for frame in frames]

        if self.transform:
            images = [self.transform(image) for image in images]

        if self.mode in ['train', 'val', 'unlabeled', 'hidden']: # you could remove unlabeled and hidden here if you did not psuedo label them
            mask_path = os.path.join(video_folder, 'mask.npy')
            mask = np.load(mask_path)
            if mask.shape[0] < len(images):
                missing_frames = len(images) - mask.shape[0]
                missing_mask = np.zeros((missing_frames, 160, 240), dtype=np.float32)
                mask = np.concatenate((mask, missing_mask), axis=0)
            last_mask = mask[-1]
            imgs = torch.stack(images)
            # output: all frames, first 11 frames, last frame, mask, last frame mask, video_folder_path
            return imgs, imgs[:11], imgs[-1], torch.tensor(mask, dtype=torch.float32), torch.tensor(last_mask, dtype=torch.float32), video_folder
        else:
            imgs = torch.stack(images)
            # output: all frames, first 11 frames, last frame
            return imgs, imgs[:11], imgs[-1], video_folder