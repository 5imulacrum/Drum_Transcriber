import argparse
import torch
from torch.utils.data import DataLoader
import yaml
import os
from .models.onset_crnn import OnsetCRNN
from .dataset import DrumDataset

def train_onset(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get('model', {}).get('device', 'cpu'))
    # Placeholder: user should prepare items list and classes
    items = []  # list of (audio_path, annotations)
    classes = ['kick', 'snare', 'hihat_closed']
    dataset = DrumDataset(items, classes, sr=cfg['audio']['sr'], hop_length=cfg['audio']['hop_length'], n_mels=cfg['audio']['n_mels'])
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = OnsetCRNN(n_mels=cfg['audio']['n_mels']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(1):
        model.train()
        for mel, labels in loader:
            mel = mel.to(device)
            # labels: (B, C, T) -> collapse classes to any-onset for onset training
            onset_labels = (labels.sum(dim=1) > 0).float()
            onset_pred = model(mel)
            # align shapes if needed
            if onset_pred.shape != onset_labels.shape:
                min_t = min(onset_pred.shape[1], onset_labels.shape[1])
                onset_pred = onset_pred[:, :min_t]
                onset_labels = onset_labels[:, :min_t]
            loss = loss_fn(onset_pred, onset_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # Save checkpoint
    torch.save(model.state_dict(), 'onset_crnn.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml'))
    args = parser.parse_args()
    train_onset(args.config)
