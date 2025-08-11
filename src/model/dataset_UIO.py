import torch
torch.autograd.set_detect_anomaly(True)
import yaml
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class UIONetDatasetV2(torch.utils.data.Dataset):
    def __init__(self, file_list, scaler=None):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True)

        features = data["features"]
        labels = data["labels"]

        delta_f_state = features[:, 9]
        Tw_state = features[:, 8]
        ax_state = features[:, 6]
        miu_state = features[:, 7]

        features = np.delete(features, [3, 4, 5, 6, 7, 8, 9], axis=1)

        features = torch.from_numpy(features).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

        delta_f_state = torch.from_numpy(delta_f_state).float().to(device)
        Tw_state = torch.from_numpy(Tw_state).float().to(device)
        ax_state = torch.from_numpy(ax_state).float().to(device)
        miu_state = torch.from_numpy(miu_state).float().to(device)

        return features, labels, delta_f_state, Tw_state, ax_state, miu_state


string_to_dataset = {
    "UIONet": UIONetDatasetV2
}
