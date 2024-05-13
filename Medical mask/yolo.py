import torch
from torch.utils.data import DataLoader
from yolov5.datasets import CSVDataset
from yolov5.models import Model
from yolov5.utils import train

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset and dataloader
dataset = CSVDataset("MyTrain.csv", "images", device)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

# Define model
model = Model("yolov5s.yaml", nc=2).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
train(model, dataloader, optimizer, device, 100)

# Save model
torch.save(model.state_dict(), "yolov5_custom.pt")