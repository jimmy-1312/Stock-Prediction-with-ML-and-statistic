"""
To be improved:
(1) Make the stock_data.csv more clear(add head,remove the left most column)
"""

import torch
import torch.nn as nn

#data shape(B,L,I) in pytorch tensor
def train(model,dataloader,device,EPOCHES):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.MSELoss()
    epoch_iterations= len(dataloader)
    loss_history = []

    for epoch in range(EPOCHES):
        
        epoch_loss = 0

        for i,batch in enumerate(dataloader):
            
            batch = [r.to(device) for r in batch]
            x, y_real = batch

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out,y_real)

            loss.backward()

            print(loss.item())

            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss/epoch_iterations
        loss_history.append(avg_epoch_loss)
            
        print(f'epoch = {epoch}, loss = {avg_epoch_loss}')
    
    return loss_history