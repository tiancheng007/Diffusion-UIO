import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random
from tqdm import tqdm

from diffusion_model import EstDiff


class QuadDataset(Dataset):
    def __init__(self, data_path, seq_len=64):
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                file_paths = sorted(glob.glob(os.path.join(data_path, "*.pt")))
            else:
                file_paths = [data_path]
        else:
            file_paths = data_path

        self.data_segments = []
        self.valid_indices = []
        self.seq_len = seq_len

        print(f"Loading dataset from {len(file_paths)} files...")
        for seg_idx, file_path in enumerate(file_paths):
            segment = torch.load(file_path)
            if len(segment) >= self.seq_len:
                self.data_segments.append(segment)
                max_end = len(segment) - self.seq_len
                min_start = self.seq_len - 1
                for start_idx in range(min_start, max_end + 1):
                    self.valid_indices.append((seg_idx, start_idx))

        print(f"Dataset loaded with {len(self.valid_indices)} valid samples")
        if not self.valid_indices:
            raise ValueError("Not enough data segments with required length")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seg_idx, start_idx = self.valid_indices[idx]
        segment = np.array(self.data_segments[seg_idx])

        seq_h = segment[start_idx:start_idx + self.seq_len]
        seq_s = segment[(start_idx - self.seq_len + 1):start_idx + 1]

        condition = seq_s[:, :4].copy()
        H = seq_h[:, 4:6].copy()

        return {
            'condition': condition,
            'H': H
        }


def train_diff(model, train_dataloader, val_dataloader=None, epochs=100, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    print(f"Training on {device}")

    best_val_loss = float('inf')
    best_model_state = None

    EPSILON = 0.002
    T = 1.0
    n_steps_training = 1000
    t_steps = [(EPSILON ** (1 / 7) + (i / (n_steps_training - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
               for i in range(0, n_steps_training)]

    t_indices = [min(int(t * model.num_timesteps), model.num_timesteps - 1) for t in t_steps]

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            cond = batch['condition'].float().to(device)
            cond = F.normalize(cond, p=2, dim=2, eps=1e-12)
            H = batch['H'].float().to(device)

            batch_size = H.shape[0]
            t_indices_batch = random.choices(t_indices, k=batch_size)
            t_idx = torch.tensor(t_indices_batch, device=device).long()

            noise = torch.randn_like(H)
            H_noisy = model.q_sample(H, t_idx, noise)

            pred_H = model(H_noisy, cond, t_idx)

            loss = F.mse_loss(pred_H, H)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()
            total_train_loss += current_loss
            batch_count += 1

            progress_bar.set_postfix({
                'batch': f"{batch_count}/{len(train_dataloader)}",
                'loss': f"{current_loss:.6f}",
                'avg_loss': f"{total_train_loss / batch_count:.6f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = total_train_loss / batch_count

        if val_dataloader is not None and (epoch + 1) % 1 == 0:
            model.eval()
            total_val_loss = 0
            val_batch_count = 0

            print("Validating...")
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    cond = batch['condition'].float().to(device)
                    cond = F.normalize(cond, p=2, dim=2, eps=1e-12)
                    H = batch['H'].float().to(device)

                    pred_H = model.predict(cond, num_samples=cond.shape[0], steps=10)
                    val_loss = F.mse_loss(pred_H, H)

                    total_val_loss += val_loss.item()
                    val_batch_count += 1

            avg_val_loss = total_val_loss / val_batch_count
            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                model_path = 'diff_model_heun_v5_seq15.pth'
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            print(f'Epoch {epoch + 1}/{epochs} | Average Train Loss: {avg_train_loss:.6f}')

            if epoch % 1 == 0:
                model_path = f'diff_model_heun_v5_seq15_epoch_{epoch + 1}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"Checkpoint saved to {model_path}")

        scheduler.step()

    if val_dataloader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    model = EstDiff(model_dim=2, cond_dim=4, seq_len=15, num_timesteps=1000)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    root_path = r"D:\0_workspace\Diff_UIO_Repo\data\diffusion_data"
    path1 = os.path.join(root_path, 'data_auto_008_0_106_train_diff_085mIz.pt')
    path2 = os.path.join(root_path, 'data_auto_008_0_106_train_diff_110mIz.pt')
    path3 = os.path.join(root_path, 'data_sharp_022_25_80_train_diff_085mIz.pt')
    path4 = os.path.join(root_path, 'data_sharp_022_25_80_train_diff_110mIz.pt')
    path5 = os.path.join(root_path, 'data_smooth_004_0_55_train_diff_085mIz.pt')
    path6 = os.path.join(root_path, 'data_smooth_004_0_55_train_diff_110mIz.pt')
    path7 = os.path.join(root_path, 'data_smooth_004_115_165_train_diff_085mIz.pt')
    path8 = os.path.join(root_path, 'data_smooth_004_115_165_train_diff_110mIz.pt')

    path9 = os.path.join(root_path, 'data_sharp_022_0_25_test_diff_110mIz.pt')
    path10 = os.path.join(root_path, 'data_smooth_004_55_115_test_diff_110mIz.pt')
    path11 = os.path.join(root_path, 'data_sharp_022_0_25_test_diff_085mIz.pt')
    path12 = os.path.join(root_path, 'data_smooth_004_55_115_test_diff_085mIz.pt')

    train_file_list = [path1, path2, path3, path4, path5, path6, path7, path8]
    val_file_list = [path9, path10, path11, path12]

    train_dataset = QuadDataset(
        train_file_list,
        seq_len=15
    )

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print(f"Train dataloader created with {len(train_dataset)} samples, {len(train_dataloader)} batches")

    val_dataset = QuadDataset(
        val_file_list,
        seq_len=15
    )

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Validation dataloader created with {len(val_dataset)} samples, {len(val_dataloader)} batches")

    model = train_diff(model, train_dataloader, val_dataloader, epochs=20, lr=2e-4)

    print(f"Training complete.")