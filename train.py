import wandb
def train(factor_model, dataloader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    for char, returns in dataloader:
        inputs = char[:,:,:-1].to(device)
        labels = returns.to(device)
        optimizer.zero_grad()
        loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
        total_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader.dataset)
    wandb.log({"Train Loss": avg_loss})
    return avg_loss


def validate(factor_model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with torch.no_grad():
        for char, returns in dataloader:
            inputs = char[:,:,:-1].to(device)
            labels = returns.to(device)
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    wandb.log({"Validation Loss": avg_loss})
    return avg_loss

def test(factor_model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with torch.no_grad():
        for char, returns in dataloader:
            inputs = char[:,:,:-1].to(device)
            labels = returns.to(device)
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    wandb.log({"Test Loss": avg_loss})
    return avg_loss

def run(factor_model, train_loader, val_loader, test_loader, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="FactorVAE", name="replicate")
    factor_model.to(device)
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(factor_model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss = train(factor_model, train_loader, optimizer)
        val_loss = validate(factor_model, val_loader)
        test_loss = test(factor_model, test_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(factor_model.state_dict(), 'best_model.pt')
    wandb.finish()
