import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import time

class SAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.in_bias = torch.nn.Parameter(torch.randn(input_size) * 0.01)
        self.encoder = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        return self.decoder(encoded), encoded


if __name__ == "__main__":
    num_epochs = 100
    l1_factor = 0.001
    lr = 0.002
    model_dim = 768
    expansion_factor = 32
    batch_size = 8192 * 2
    weight_decay = 0.0001

    embeds = torch.cat(torch.load("all_embeds.pt"))
    sae = SAE(model_dim, model_dim * expansion_factor).to("cuda")
    sae_forward = torch.compile(sae)

    dataset = TensorDataset(embeds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False)
    pbar = tqdm(range(len(dataloader) * num_epochs))

    sae_opt = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(sae_opt, max_lr=lr, total_steps=len(dataloader) * num_epochs, pct_start=0.3)

    mse_losses = []
    l1_losses = []
    for epoch in range(100):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False)
        for batch in dataloader:
            with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch = batch[0].to("cuda")
                out, encoded = sae_forward(batch)
                mse_loss = torch.nn.functional.mse_loss(out, batch)
                mse_losses.append(mse_loss.item())
                encoded_l1 = torch.nn.functional.l1_loss(encoded, torch.zeros_like(encoded))
                l1_losses.append(encoded_l1.item())
                loss = mse_loss + encoded_l1 * l1_factor
                sae_opt.zero_grad(set_to_none=True)
                loss.backward()
                sae_opt.step()
                scheduler.step()
                pbar.set_description(f"mse_loss: {mse_loss.item()}, l1_loss: {encoded_l1.item()}, epoch: {epoch}, lr: {sae_opt.param_groups[0]['lr']}")
                pbar.update(1)

    torch.save(sae.state_dict(), "sae.pt")