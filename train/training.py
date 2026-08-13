"""
To be improved:
(1) Make the stock_data.csv more clear(add head,remove the left most column)
"""

import torch
import torch.nn as nn

#data shape(B,L,I) in pytorch tensor
def train(model,dataloaders,device,EPOCHES):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(EPOCHES):
        
        epoch_loss = 0

        for dataloader in dataloaders:

            dataloader_loss = 0
            
            for i,batch in enumerate(dataloader):
                
                batch = [r.to(device) for r in batch]
                x, y_real = batch

                optimizer.zero_grad()

                out = model(x)

                loss = criterion(out,y_real)

                loss.backward()

                print(loss.item())

                optimizer.step()

                dataloader_loss += loss.item()

            dataloader_loss = dataloader_loss/len(dataloader)
            epoch_loss += dataloader_loss

        loss_history.append(epoch_loss) #here epoch loss is linear with size of dataloaders, you may consider divide it by `len(dataloaders)`
            
        print(f'epoch = {epoch}, loss = {epoch_loss}')
    
    return loss_history