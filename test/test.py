import sys
sys.path.append("./")
from model.LSTM import LSTM
from get_data import get_simulated_data
import torch
import matplotlib.pyplot as plt

model = LSTM(1,10)
model.load_state_dict(torch.load("./model_save/lstm2.pth"))

real_pattern = get_simulated_data.generate_pattern(100,difficulty="easy")
data_with_noises = get_simulated_data.generate_data(100,1,"easy")
data_with_noises = list(data_with_noises)[:100]
plt.plot(real_pattern, 'b')
with torch.no_grad():
    # for i in range(50):
    #     out = model(torch.tensor(data_with_noises,dtype=torch.float32).view(1,-1,1))
    #     print(out[0,-1,0])
    #     data_with_noises.append(out.detach().cpu().numpy()[0,-1,0])
    out = model(torch.tensor(data_with_noises,dtype=torch.float32).view(1,-1,1))
    out = out.detach().cpu().numpy()
plt.plot(data_with_noises, 'r')
plt.plot(out[0,:,0],'g')
print(data_with_noises)
print(out)
plt.savefig("./test/lstm2.png")
plt.show()
