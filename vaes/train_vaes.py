from worldmodels.data.dataset_structures import DataRecord
import worldmodels.vaes.vaes as vaes

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

SAVE_VAE = True
agent_name = 'robot'
sensor_name = 'IR_1'

p_file = "../data/recordings/environment_soutenance/IR_data_fov270_1080dots_60000stepsMultiStep=2_soutenance_scale13E-1.p"
train_data = DataRecord(p_file, train=True, train_rate=0.8, step_sample=1, step_dot=1)
train_loaded = DataLoader(train_data, batch_size=64, shuffle=True)

eval_data = DataRecord(p_file, train=False, train_rate=0.8, step_sample=1, step_dot=1)
eval_loaded = DataLoader(eval_data, batch_size=64, shuffle=True)

encoder = vaes.VAELow()

optimizer = optim.Adam(encoder.parameters(), lr=0.007)
scheduler = StepLR(optimizer, step_size=3, gamma=0.7)

nb_epochs = 25
for epoch in range(1, nb_epochs + 1):
    vaes.train_vae(encoder, optimizer, epoch, train_loaded, agent_name, sensor_name)
    scheduler.step()

vaes.val_vae(encoder, eval_loaded, agent_name, sensor_name)

plt.plot(vaes.losses_training)
if SAVE_VAE:
    name = "VAELow"
    torch.save({'state_dict': encoder.state_dict()}, "trained_vaes/"+name+".pt")
    plt.savefig("trained_vaes/training_loss/"+name)
plt.show()
