from tqdm import tqdm
from UIO.ss_model import UIO_Model
from UIO.filtering import Unknown_Input_Observer
import cfgs.config as config
from dataset_UIO import string_to_dataset
import torch
import numpy as np
import os
import time


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def compute_percent_error(predicted, target):
    percent_errors = dict()
    for key in predicted.keys():
        if target.get(key):
            percent_errors[key] = np.abs(predicted[key] - target[key]) / target[key] * 100
    return percent_errors

        

def evaluate_predictions(test_data_loader):
    test_losses1 = []
    test_losses2 = []
    inference_times = []
    loss_fn = torch.nn.MSELoss(reduction='mean')
    max_errors = [0.0]
    file2 = open(r"D:\0_workspace\Diff_UIO_Repo\delta_f_pred.txt", "w",
                encoding='utf-8')
    file3 = open(r"D:\0_workspace\Diff_UIO_Repo\delta_f_gt.txt", "w",
                encoding='utf-8')
    file4 = open(r"D:\0_workspace\Diff_UIO_Repo\vy_pred.txt", "w",
                encoding='utf-8')
    file5 = open(r"D:\0_workspace\Diff_UIO_Repo\vy_gt.txt", "w",
                encoding='utf-8')
    file6 = open(r"D:\0_workspace\Diff_UIO_Repo\inf_time.txt", "w",
                encoding='utf-8')

    UIO_filter = Unknown_Input_Observer(UIO_Model(mode='test'))
    path_dir = r'D:\0_workspace\Diff_UIO_Repo\src\model\diffusion'
    model_path = os.path.join(path_dir, "diff_model_heun_v5_seq15_best.pth")
    model = UIO_filter.gen_net.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for inp_val, lab_val, delta_f_inp_val, _, _, _ in test_data_loader:
            UIO_filter.reset(clean_history=True)
            inp_val = inp_val.to(device)
            lab_val = lab_val.to(device)
            vy_inp_val = lab_val[:, 0].to(device)
            delta_f_inp_val = delta_f_inp_val.to(device)
            val_target_states = vy_inp_val

            for k in tqdm(range(1, inp_val.shape[0])):
                start = time.time()
                vy_pred, delta_f = UIO_filter.filtering_e2e_diff(inp_val[k].reshape((-1, 1)), k)
                end = time.time()
                ext_time = end - start
                inference_times.append(end - start)
                test_loss1 = loss_fn(vy_pred, vy_inp_val[k].reshape(-1))
                test_loss2 = loss_fn(delta_f, delta_f_inp_val[k-1].reshape(-1))
                test_losses1.append(test_loss1.cpu().detach().numpy())
                test_losses2.append(test_loss2.cpu().detach().numpy())

                file2.write(str(delta_f.cpu().detach().numpy()[0]) + '\n')
                file3.write(str(delta_f_inp_val[k-1].cpu().detach().numpy()) + '\n')
                file4.write(str(vy_pred.cpu().detach().numpy()[0]) + '\n')
                file5.write(str(val_target_states[k].cpu().detach().numpy()) + '\n')
                file6.write(str(ext_time) + '\n')

            print("Average Inference Time:", np.mean(inference_times))
            print("Vy RMSE:", np.sqrt(np.mean(test_losses1, axis=0)))
            print("Delta_f RMSE:", np.sqrt(np.mean(test_losses2, axis=0)))
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()


def collate_fn(batch):
    features, labels, delta_f_state, Tw_state, ax_state, miu_state = batch[0]
    return features, labels, delta_f_state, Tw_state, ax_state, miu_state


if __name__ == "__main__":
    args = config.general_settings()
    argdict: dict = vars(args)


    args.use_cuda = True  # use GPU or not

    root_path = r"D:\0_workspace\Diff_UIO_Repo\data"

    path6 = os.path.join(root_path, 'data_sharp_022_0_25_test_h1.npz')
    path7 = os.path.join(root_path, 'data_extr_024b_0_40_test_h1.npz')
    path8 = os.path.join(root_path, 'data_smooth_004_55_115_test_h1.npz')


    test_file_list = [path8]

    test_dataset = string_to_dataset["UIONet"](test_file_list)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    basename = os.path.basename(test_file_list[0])
    file_name = os.path.splitext(basename)[0]

    evaluate_predictions(test_data_loader)
