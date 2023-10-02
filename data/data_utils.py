import numpy as np

def calc_class_dist(data_loader, num_tasks):
    nums_pos = np.zeros(num_tasks)
    for idx, (_, labels) in enumerate(data_loader):
        for t in range(num_tasks):
            nums_pos[t] += labels[:, t].sum()
    return nums_pos / len(data_loader.dataset)