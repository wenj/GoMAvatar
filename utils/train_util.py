import time
import numpy as np

###############################################################################
## Misc Functions
###############################################################################

def cpu_data_to_gpu(cpu_data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.cuda() for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = {sub_k: sub_val.cuda() for sub_k, sub_val in val.items()}
        else:
            gpu_data[key] = val.cuda()

    return gpu_data


###############################################################################
## Timer
###############################################################################

class Timer():
    def __init__(self):
        self.curr_time = 0

    def begin(self):
        self.curr_time = time.time()

    def log(self):
        diff_time = time.time() - self.curr_time
        self.begin()
        return f"{diff_time:.2f} sec"

class Timer():
    def __init__(self):
        self.curr_time = 0
        self.total_time = 0

        self.status = False

    def tick(self):
        assert not self.status, "timer already starts"
        self.status = True
        self.curr_time = time.time()

    def tock(self):
        assert self.status, "timer does not start"
        self.status = False
        diff_time = time.time() - self.curr_time
        self.total_time += diff_time

    def reset(self):
        self.status = False
        self.total_time = 0

    def log(self):
        return f"{self.total_time:.2f} sec"


def make_weights_for_pose_balance(dataset, n_classes=8):
    count = [0] * n_classes
    Rs = dataset.get_all_Es()[:, :3, :3]
    Rs = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ])[None] @ Rs
    yaws = np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])
    # rolls = np.arctan2(Rs[:, 2, 1], Rs[:, 2, 2])
    # pitchs = np.arctan2(-Rs[:, 2, 0], np.sqrt(Rs[:, 2, 1] ** 2 + Rs[:, 2, 2] ** 2))
    #
    # idx = [4561, 4757, 4784, 2796]
    #
    yaws_deg = yaws / np.pi * 180
    # rolls_deg = rolls / np.pi * 180
    # pitchs_deg = pitchs / np.pi * 180
    #
    # import pdb; pdb.set_trace()
    bin_ids = ((yaws + np.pi) / (np.pi * 2) * n_classes).astype(int)
    bin_ids = np.clip(bin_ids, a_max=n_classes - 1, a_min=None)
    for i in range(n_classes):
        count[i] = np.sum(bin_ids == i)
    count = sum(count) / np.clip(count, a_min=10, a_max=None)
    weights = np.array(count)[bin_ids]
    return weights
