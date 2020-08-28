from worldmodels.data.dataset_structures import DataRecord, IRRecordSequences
import worldmodels.vaes.vaes as vaes
import worldmodels.rnns.gru as gru

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

SAVE_GRU = True
agent_name = 'robot'
sensor_name = 'IR_1'
# data_vae = DataRecord("../data/recordings/IR_data_fov270_1080dots_60000steps.p", train=True, train_rate=0.8,
#                         step_sample=1, step_dot=1)
# print(data_vae.minimaxis['states']['agent_name']['sensor_name'])
min_observation_data_VAE = 0.032
max_observation_data_VAE = 1

p_file = "../data/recordings/new_environment/IR_data_fov270_1080dots_60000stepsMultiStep=2_scale125E-2.p"
train_IRdata = IRRecordSequences(p_file, min_observation_data_VAE, max_observation_data_VAE, train=True,
                                 seq_in=16, seq_out=1, agent_name=agent_name)
train_IRdata_loaded = DataLoader(train_IRdata, batch_size=64, shuffle=True, drop_last=True)

eval_IRdata = IRRecordSequences(p_file, min_observation_data_VAE, max_observation_data_VAE, train=False,
                                seq_in=16, seq_out=1, agent_name=agent_name)
eval_IRdata_loaded = DataLoader(eval_IRdata, batch_size=64, shuffle=True, drop_last=True)

encoder = vaes.VAELow()

name_encoder = "VAELow_25epochs_60000steps_NEW_ENVIRONMENT_SCALE125E-2__batch64_Adam_lr7E-3"
state_dict_vae = torch.load("../vaes/trained_vaes/" + name_encoder + ".pt")['state_dict']
encoder.load_state_dict(state_dict_vae)

RNN = gru.GRUNet(seq_in=16, seq_out=1, n_layers=1, drop_prob=0.0, input_action_dim=3, input_states_dim=32, output_dim=32)

optimizer = optim.Adam(RNN.parameters(), lr=0.0005)

scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

nb_epochs = 15
for epoch in range(1, nb_epochs + 1):
    gru.train_gru(RNN, encoder, optimizer, epoch, train_IRdata_loaded)
    scheduler.step()

gru.val_gru(RNN, encoder, eval_IRdata_loaded)

plt.plot(gru.losses_training)
if SAVE_GRU:
    name = "GRU_seqin16_seqout3_1layer_15epochs_60000steps_MultiStep2_SCALE125E-2_batch128_Adam_lr5E-4"
    torch.save({'state_dict': RNN.state_dict()}, "trained_rnns/"+name+".pt")
    plt.savefig("trained_rnns/training_loss/"+name)
plt.show()
