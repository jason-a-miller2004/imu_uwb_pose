# tsne_participants.py
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for SSH/headless
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from imu_uwb_pose.config import config
from imu_uwb_pose.training.footposer_tsne_dataset import footposer_tsne_dataset
from imu_uwb_pose.training.imu_uwb_pose_model import imu_uwb_pose_model as model
from imu_uwb_pose.training.utils import pad_seq_tsne

parser = argparse.ArgumentParser(description="t-SNE plotting for FootPoser embeddings")
parser.add_argument("--experiment", required=True, help="Experiment folder name used for checkpoints and figs")
parser.add_argument("--name", required=True, help="Run/fold name (e.g., held-out participant)")
args = parser.parse_args()

experiment = args.experiment
name = args.name
config = config(experiment=experiment, name=name, dataset='footposer_tsne_dataset')
device = config.device

# Directory to save figures
out_dir = config.fig_path / experiment / name
out_dir.mkdir(exist_ok=True, parents=True)

# fill these from your training config
best_model_txt_path = os.path.join(config.checkpoint_path, "best_model.txt")
with open(best_model_txt_path, "r") as f:
    lines = f.readlines()
best_model_path = lines[0].strip()
model = model.load_from_checkpoint(
    best_model_path,
    map_location=config.device,
    config=config
)
model.eval()
lstm = model.model
dataloader = DataLoader(footposer_tsne_dataset(config), batch_size=config.batch_size, shuffle=False, collate_fn=pad_seq_tsne)
# 2) Collect embeddings across all participants (ideally from the held-out fold or from all folds)
embeds, pids,motion_ids = [], [], []

# It should yield: batch_x: (B,T,INPUT_DIM), batch_lens: list[int], batch_pid: (B,) participant ids
for batch_x, batch_lens, batch_pid, batch_motion_id in dataloader:
    batch_x = batch_x.to(device)
    with torch.no_grad():
        emb, _, _ = lstm.encode(batch_x, batch_lens)  # (B, H*dirs)
    embeds.append(emb.cpu().numpy())
    pids.append(np.asarray(batch_pid))
    motion_ids.append(np.asarray(batch_motion_id))

X = np.concatenate(embeds, axis=0)          # (N, D)
participant_labels = np.concatenate(pids, axis=0)       # (N,)
motion_labels = np.concatenate(motion_ids, axis=0)  # (N,)

# 4) t-SNE (tune perplexity to sample count; 5–50 typical)
tsne = TSNE()
Z = tsne.fit_transform(X)

# 5) Two plots: by participant (sid) and by motion_id
def plot_tsne(Z, labels, title, prefix, outfile):
    plt.figure(figsize=(7, 6))
    uvals = np.unique(labels)
    # Stable palette that cycles if more than 20 classes
    cmap = plt.get_cmap("tab20")
    color_of = {u: cmap(i % cmap.N) for i, u in enumerate(uvals)}
    # points
    for u in uvals:
        mask = labels == u
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, label=f"{prefix}{int(u)}", c=[color_of[u]])
   
    plt.title(title)
    plt.legend(markerscale=2, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()

# Cast to ints for cleaner legend labels
labels_participant = participant_labels.astype(int)
labels_motion = motion_labels.astype(int)

plot_tsne(
    Z,
    labels_participant,
    "t-SNE of sequence embeddings (colored by participant)",
    "P",
    os.path.join(out_dir, "tsne_by_participant.png"),
)
plot_tsne(
    Z,
    labels_motion,
    "t-SNE of sequence embeddings (colored by motion)",
    "M",
    os.path.join(out_dir, "tsne_by_motion.png"),
)
print(f"Saved: {os.path.join(out_dir, 'tsne_by_participant.png')}")
print(f"Saved: {os.path.join(out_dir, 'tsne_by_motion.png')}")