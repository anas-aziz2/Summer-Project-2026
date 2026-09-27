import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .score import ScoreNetwork


def train_model(
    configs_tensor,
    sde,
    optimizer_name="Adam",
    lr=1e-4,
    batch_size=100,
    width=64,
    depth=3,
    num_epochs=400,
    log_every=10,
    grad_clip=1.0,
):
    dataset = TensorDataset(configs_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ScoreNetwork(hidden_dim=width, num_layers=depth)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for (x_batch,) in loader:
            bsz = x_batch.shape[0]

            t = torch.rand(bsz, 1)
            sigma = sde.sigma(t)
            sigma_expanded = sigma.view(-1, 1, 1)

            z = torch.randn_like(x_batch)
            z = z - z.mean(dim=1, keepdim=True)

            x_noisy = x_batch + sigma_expanded * z
            log_sigma = torch.log(sigma)
            target = -z

            f_pred = model(x_noisy, log_sigma)
            loss_per_sample = ((f_pred - target) ** 2).sum(dim=(1, 2))
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | Loss = {avg_loss:.6f}")

    return model, loss_history