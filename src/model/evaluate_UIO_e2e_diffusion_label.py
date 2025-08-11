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

        

def evaluate_predictions(test_data_loader, save_path_name):
    test_losses1 = []
    test_losses2 = []

    inference_times = []
    loss_fn = torch.nn.MSELoss(reduction='mean')

    UIO_filter = Unknown_Input_Observer(UIO_Model(mode='test'))
    path_dir = r'D:\0_workspace\Diff_UIO_Repo\output\UIONet_LPV_diffusion_label_085m_Iz_path1_4_run2'
    model_path = os.path.join(path_dir, "epoch_best.pth")
    model = UIO_filter.uncertain_net.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for inp_val, lab_val, delta_f_inp_val, _, _, _ in test_data_loader:
            UIO_filter.reset(clean_history=True)
            inp_val = inp_val.to(device)
            lab_val = lab_val.to(device)
            vy_inp_val = lab_val[:, 0].to(device)
            delta_f_inp_val = delta_f_inp_val.to(device)

            data_list = []

            for k in tqdm(range(1, inp_val.shape[0])):
                start = time.time()
                vy_pred, delta_f, H_est = UIO_filter.filtering_e2e_label(inp_val[k].reshape((-1, 1)))
                end = time.time()
                vx_obs = inp_val[k][0]
                r_obs = inp_val[k][1]
                diffusion_label_data = torch.tensor([
                    float(vx_obs),
                    float(r_obs),
                    float(vy_pred),
                    float(delta_f),
                    float(H_est[0]),
                    float(H_est[1])
                ])
                data_list.append(diffusion_label_data)
                inference_times.append(end - start)
                test_loss1 = loss_fn(vy_pred, vy_inp_val[k].reshape(-1))
                test_loss2 = loss_fn(delta_f, delta_f_inp_val[k-1].reshape(-1))
                test_losses1.append(test_loss1.cpu().detach().numpy())
                test_losses2.append(test_loss2.cpu().detach().numpy())


            torch.save(data_list, save_path_name)
            print(np.array(data_list).shape)

            print("Average Inference Time:", np.mean(inference_times))
            print("Vy RMSE:", np.sqrt(np.mean(test_losses1, axis=0)))
            print("Delta_f RMSE:", np.sqrt(np.mean(test_losses2, axis=0)))

def collate_fn(batch):
    features, labels, delta_f_state, Tw_state, ax_state, miu_state = batch[0]
    return features, labels, delta_f_state, Tw_state, ax_state, miu_state


if __name__ == "__main__":
    args = config.general_settings()
    argdict: dict = vars(args)


    args.use_cuda = True  # use GPU or not

    root_path = r"D:\0_workspace\Diff_UIO_Repo\data"
    path1 = os.path.join(root_path, 'data_smooth_008_0_106_train_h1.npz')
    path2 = os.path.join(root_path, 'data_sharp_022_25_80_train_h1.npz')
    path3 = os.path.join(root_path, 'data_smooth_004_0_55_train_h1.npz')
    path4 = os.path.join(root_path, 'data_smooth_004_115_165_train_h1.npz')

    path6 = os.path.join(root_path, 'data_sharp_022_0_25_test_h1.npz')
    path7 = os.path.join(root_path, 'data_extr_024b_0_40_test_h1.npz')
    path8 = os.path.join(root_path, 'data_smooth_004_55_115_test_h1.npz')

    test_file_list = [path3]
    save_path = os.path.join(root_path, 'diffusion_data/data_smooth_004_0_55_train_diff_085mIz.pt')

    test_dataset = string_to_dataset["UIONet"](test_file_list)


    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    basename = os.path.basename(test_file_list[0])
    file_name = os.path.splitext(basename)[0]

    evaluate_predictions(test_data_loader, save_path)
