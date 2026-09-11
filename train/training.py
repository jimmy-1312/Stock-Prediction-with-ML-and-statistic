"""
To be improved:
(1) Make the stock_data.csv more clear(add head,remove the left most column)
"""

import torch
import torch.nn as nn

#data shape(B,L,I) in pytorch tensor
def train(model,dataloaders,device,EPOCHES,lstm_mode):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    if lstm_mode == "value":
        criterion = nn.MSELoss()
    elif lstm_mode == "category":
        criterion = nn.NLLLoss()
        log_softmax = nn.LogSoftmax(dim=1)
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
                if lstm_mode == "category":
                    out = log_softmax(out)

                loss = criterion(out,y_real)

                loss.backward()

                optimizer.step()

                dataloader_loss += loss.item()

            dataloader_loss = dataloader_loss/len(dataloader)
            epoch_loss += dataloader_loss

        loss_history.append(epoch_loss) #here epoch loss is linear with size of dataloaders, you may consider divide it by `len(dataloaders)`
            
        print(f'epoch = {epoch}, loss = {epoch_loss}')
    
    return loss_history