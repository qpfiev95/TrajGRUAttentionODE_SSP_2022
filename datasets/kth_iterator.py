import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import datasets.vtrans as vtransforms


class Dataset_base(Dataset):
    def __init__(self, train=True, seq_len=30, phase="train"):

        # Get options
        self.window_size = seq_len
        self.sample_size = seq_len
        self.phase = phase
        self.irregular = False
        self.extrap = True
        self.train = train
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Print out Dataset setting
        #regularity = "irregular" if self.opt.irregular else "regular"
        #task = "extrapolation" if self.opt.extrap else "interpolation"
        #print(f"[Info] Dataset:{self.opt.dataset} / regularity:{regularity} / task:{task}")

    def sample_regular_interp(self, images):
        seq_len = images.shape[0]
        assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"

        win_start = np.random.randint(0, seq_len - self.sample_size + 1) if self.train else 0

        if self.phase == 'train':
            input_images = images[np.arange(win_start, win_start + self.sample_size, 2), ...]
            mask = torch.ones((self.sample_size // 2, 1))
        else:
            input_images = images[win_start: win_start + self.sample_size]
            mask = torch.zeros((self.sample_size, 1))
            mask[np.arange(0, self.sample_size, 2), :] = 1

        mask = mask.type(torch.FloatTensor).to(self.device)
        return input_images, mask

    def sample_regular_extrap(self, images):
        """ Same as sample_regular_interp, may be different when utils.sampling """
        seq_len = images.shape[0]
        assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
        # win_start = random.randint(0, seq_len - self.sample_size - 1) if self.train else 0
        win_start = random.randint(0, seq_len - self.sample_size) if self.train else 0
        input_images = images[win_start: win_start + self.sample_size]
        mask = torch.ones((self.sample_size, 1))
        mask = mask.type(torch.FloatTensor).to(self.device)
        return input_images, mask

    def sample_irregular_interp(self, images):
        seq_len = images.shape[0]
        if seq_len <= self.window_size:
            assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
            win_start = 0
            rand_idx = sorted(
                np.random.choice(list(range(win_start + 1, seq_len - 1)), size=self.sample_size - 2, replace=False))
            rand_idx = [win_start] + rand_idx + [seq_len - 1]
        elif seq_len > self.window_size:
            win_start = random.randint(0, seq_len - self.window_size - 1) if self.train else 0
            rand_idx = sorted(np.random.choice(list(range(win_start + 1, win_start + self.window_size - 1)),
                                               size=self.sample_size - 2, replace=False))
            rand_idx = [win_start] + rand_idx + [win_start + self.window_size - 1]
        # [Caution]: Irregular setting return window-sized images and it is filtered out
        # Sample images
        input_idx = list(range(win_start, win_start + self.window_size))
        input_images = images[input_idx]
        # Sample masks
        mask = torch.zeros((self.window_size, 1))
        mask_idx = [r - win_start for r in rand_idx]
        mask[mask_idx, :] = 1
        mask = mask.type(torch.FloatTensor).to(self.device)
        return input_images, mask

    def sample_irregular_extrap(self, images):
        seq_len = images.shape[0]
        assert self.window_size % 2 == 0, "[Error] window_size should be even number"
        assert self.sample_size % 2 == 0, "[Error] sample_size should be even number"
        half_window_size = self.window_size // 2
        half_sample_size = self.sample_size // 2
        if seq_len <= self.window_size:
            assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
            win_start = 0  # + 1
            half_window_size = seq_len // 2
            rand_idx_in = sorted(
                np.random.choice(list(range(win_start + 1, win_start + half_window_size)), size=half_sample_size - 1,
                                 replace=False))
            rand_idx_out = sorted(np.random.choice(list(range(win_start + half_window_size, win_start + seq_len - 1)),
                                                   size=half_sample_size - 1, replace=False))
            rand_idx = [win_start] + rand_idx_in + rand_idx_out + [win_start + seq_len - 1]
        elif seq_len > self.window_size:
            win_start = random.randint(0, seq_len - self.window_size - 1) if self.train else 0  # + 1
            rand_idx_in = sorted(
                np.random.choice(list(range(win_start + 1, win_start + half_window_size)), size=half_sample_size - 1,
                                 replace=False))
            rand_idx_out = sorted(
                np.random.choice(list(range(win_start + half_window_size, win_start + self.window_size - 1)),
                                 size=half_sample_size - 1, replace=False))
            rand_idx = [win_start] + rand_idx_in + rand_idx_out + [win_start + self.window_size - 1]
        # [Caution]: Irregular setting return window-sized images and it is filtered out
        # Sample images
        input_idx = list(range(win_start, win_start + self.window_size))
        input_images = images[input_idx]
        # Sample mask
        mask = torch.zeros((self.window_size, 1))
        mask_idx = [r - win_start for r in rand_idx]
        mask[mask_idx, :] = 1
        return input_images, mask

    def sampling(self, images):
        # Sampling
        if not self.irregular and not self.extrap:
            input_images, mask = self.sample_regular_interp(images=images)
        elif not self.irregular and self.extrap:
            input_images, mask = self.sample_regular_extrap(images=images)
        elif self.irregular and not self.extrap:
            input_images, mask = self.sample_irregular_interp(images=images)
        else:
            input_images, mask = self.sample_irregular_extrap(images=images)
        return input_images, mask

def remove_files_under_sample_size(image_path, threshold):
    temp_image_list = [x for x in os.listdir(image_path)]
    image_list = []
    remove_count = 0
    for i, file in enumerate(temp_image_list):
        _image = np.load(os.path.join(image_path, file))
        if _image.shape[0] >= threshold:
            image_list.append(file)
        else:
            remove_count += 1
    if remove_count > 0:
        print(f"Remove {remove_count:03d} shorter than than sample_size...")
    return image_list

class VideoDataset(Dataset_base):

    def __init__(self, dataset="kth", train=True):
        super(VideoDataset, self).__init__(train=train)
        # Dataroot & Transform
        self.data_root = './kth/'
        self.train = train
        vtrans = [vtransforms.CenterCrop(size=120), vtransforms.Scale(size=128)]

        if self.train:
            vtrans += [vtransforms.RandomHorizontalFlip()]
            vtrans += [vtransforms.RandomRotation()]
        vtrans += [vtransforms.ToTensor(scale=True)]
        #vtrans += [vtransforms.Normalize(0.5, 0.5)] if opt.input_norm else []
        self.vtrans = T.Compose(vtrans)

        if self.train:
            self.image_path = os.path.join(self.data_root, 'train')
        else:
            self.image_path = os.path.join(self.data_root, 'test')
        #threshold = self.window_size
        self.image_list = os.listdir(self.image_path)
        self.image_list = sorted(self.image_list)

    def __getitem__(self, index):
        assert self.sample_size <= self.window_size, "[Error] sample_size > window_size"
        images = np.load(os.path.join(self.image_path, self.image_list[index]))
        # Sampling
        input_images, mask = self.sampling(images=images)
        # Transform
        input_images = self.vtrans(input_images)  # return (b, c, h, w)
        return input_images, mask

    def __len__(self):
        return len(self.image_list)

def split_data_extrap(data_dict):
    n_observed_tp = data_dict["data"].size(1) // 2
    split_dict = {"observed_data": data_dict["data"][:, :n_observed_tp, :].clone(),
                  "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
                  "data_to_predict": data_dict["data"][:, n_observed_tp:, :].clone(),
                  "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone(),
                  "observed_mask": None, "mask_predicted_data": None}
    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()
    split_dict["mode"] = "extrap"
    return split_dict

def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict

def split_and_subsample_batch(data_dict, data_type="train", extrap=True):
    if data_type == "train":
        # Training set
        if extrap:
            processed_dict = split_data_extrap(data_dict)
        #else:
        #    processed_dict = split_data_interp(data_dict, opt)
    else:
        # Test set
        if extrap:
            processed_dict = split_data_extrap(data_dict)
        #else:
        #    processed_dict = split_data_interp(data_dict, opt)
    # add mask
    processed_dict = add_mask(processed_dict)
    return processed_dict

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def parse_datasets(device, batch_size, sample_size):
    def video_collate_fn(batch, time_steps, data_type="train"):
        images = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        data_dict = {"data": images, "time_steps": time_steps, "mask": mask}
        data_dict = split_and_subsample_batch(data_dict, data_type=data_type, extrap=True)
        data_dict['mode'] = data_type
        return data_dict
    #################################
    #if opt.irregular:
    #    time_steps = np.arange(0, opt.window_size) / opt.window_size
    #else:
    #    if opt.extrap:
    #        time_steps = np.arange(0, opt.sample_size) / opt.sample_size
    #    else:
    #        time_steps = np.arange(0, opt.sample_size // 2) / (opt.sample_size // 2)
            # time_steps = np.arange(0, opt.sample_size) / opt.sample_size
    time_steps = np.arange(1, sample_size+1)
    time_steps = torch.from_numpy(time_steps).type(torch.FloatTensor).to(device)
    train_dataloader = DataLoader(VideoDataset(dataset="kth",train=True),
                                  batch_size=4,
                                  shuffle=True,
                                  collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="train"))
    test_dataloader = DataLoader(VideoDataset(dataset="kth", train=False),
                                 batch_size=4,
                                 shuffle=False,
                                 collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="test"))
    data_objects = {"train_dataloader": inf_generator(train_dataloader),
                    "test_dataloader": inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}
    return data_objects

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None
            }

def get_data_dict(dataloader):
    data_dict = dataloader.__next__()
    return data_dict

def get_next_batch(data_dict, test_interp=False):
    device = get_device(data_dict["observed_data"])
    batch_dict = get_dict_template()
    # preserving values:
    batch_dict["mode"] = data_dict["mode"]
    batch_dict["observed_data"] = data_dict["observed_data"]
    batch_dict["observed_tp"] = data_dict["observed_tp"]
    batch_dict["data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]
    # Input: Mask out skipped data
    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"]
        filter_mask = batch_dict["observed_mask"].unsqueeze(-1).unsqueeze(-1).to(device)
        if not test_interp:
            batch_dict["observed_data"] = filter_mask * batch_dict["observed_data"]
        else:
            selected_mask = batch_dict["observed_mask"].squeeze(-1).byte()
            b, t, c, h, w = batch_dict["observed_data"].size()
            batch_dict["observed_data"] = batch_dict["observed_data"][selected_mask, ...].view(b, t // 2, c, h, w)
            batch_dict["observed_mask"] = torch.ones(b, t // 2, 1).cuda()
    # Pred: Mask out skipped data
    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"]
        filter_mask = batch_dict["mask_predicted_data"].unsqueeze(-1).unsqueeze(-1).to(device)
        if not test_interp:
            batch_dict["orignal_data_to_predict"] = batch_dict["data_to_predict"].clone()
            batch_dict["data_to_predict"] = filter_mask * batch_dict["data_to_predict"]
        else:
            b, t, c, h, w = batch_dict["data_to_predict"].size()
            # specify times
            batch_dict["tp_to_predict"] = torch.from_numpy(np.arange(0, t) / t).type(torch.FloatTensor).cuda()
            # mask out
            selected_mask = torch.ones_like(batch_dict["mask_predicted_data"]) - batch_dict["mask_predicted_data"]
            selected_mask[:, -1, :] = 0.  # exclude last frame
            selected_mask = selected_mask.squeeze(-1).byte()
            batch_dict["mask_predicted_data"] = selected_mask
    return batch_dict

#### Testing
phase = "train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kth_iterator = parse_datasets(device=device, batch_size=4, sample_size=30)

train_dataloader = kth_iterator['train_dataloader']
test_dataloader = kth_iterator['test_dataloader']
n_train_batches = kth_iterator['n_train_batches']
n_test_batches = kth_iterator['n_test_batches']

for iter in range(n_test_batches):
    data_dict = get_data_dict(test_dataloader)
    input_sequence_1 = data_dict['observed_data']
    target_sequence_1 = data_dict['data_to_predict']
    batch_dict = get_next_batch(data_dict)
    #input_sequence_2 = batch_dict['observed_data']
    #target_sequence_2 = batch_dict['data_to_predict']
    print(iter, input_sequence_1.max(), target_sequence_1.max())
    #print(input_sequence_2.max(), target_sequence_2.max())
