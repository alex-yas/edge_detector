from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.edge_detector import UnetLikeEdgeDetector
from utilities.edge_dataset import EdgeDetectionDataset


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          loss_function: Callable, optimizer: optim.Optimizer, 
          num_epochs: int = 20) -> None:  
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, edges in train_loader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_function(outputs, edges)
            
            loss.backward()            
            optimizer.step()

            train_loss = train_loss + loss.item()
            
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)                
                val_loss += loss.item()
        
        print(f'Val Loss: {val_loss / len(val_loader):.4f}')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 
                       f"model_weights/unet_like_another{epoch}")



def main():
    train_dataset = EdgeDetectionDataset(path="data/train")
    val_dataset = EdgeDetectionDataset(path="data/valid")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    model = UnetLikeEdgeDetector()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train(
        model=model, 
        train_loader=train_loader,
        val_loader= val_loader, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        num_epochs=100
    )


if __name__ == "__main__":
    main()