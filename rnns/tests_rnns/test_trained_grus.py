from worldmodels.data.dataset_structures import DataRecord, IRRecordSequences
import worldmodels.vaes.vaes as vaes
import worldmodels.rnns.gru as gru

import numpy as np
import matplotlib.pyplot as plt
from random import randint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EVAL_RNN = False
NB_SAMPLE = 10
agent_name = 'robot'
sensor_name = 'IR_1'

# data_vae = DataRecord("../../data/recordings/IR_data_fov270_1080dots_60000stepsMultiStep=2.p", train=True, train_rate=0.8,
#                         step_sample=1, step_dot=1)
# print(data_vae.minimaxis['states'][agent_name][sensor_name])
min_observation_data_VAE = 0.032
max_observation_data_VAE = 1

p_file = "../../data/recordings/environment_soutenance/IR_data_fov270_1080dots_60000stepsMultiStep=2_soutenance_scale13E-1.p"
eval_IRdata = IRRecordSequences(p_file, min_observation_data_VAE, max_observation_data_VAE, train=False,
                                seq_in=16, seq_out=3, agent_name=agent_name)
eval_IRdata_loaded = DataLoader(eval_IRdata, batch_size=1, shuffle=True, drop_last=True)

encoder = vaes.VAELow()
name_encoder = "VAELow_25epochs_60000steps_SoutenanceSHUFFLED_SCALE13E-1__batch64_Adam_lr7E-3"
state_dict_vae = torch.load("../../vaes/trained_vaes/soutenance/" + name_encoder + ".pt")['state_dict']
encoder.load_state_dict(state_dict_vae)

rnn = gru.GRUNet(seq_in=16, seq_out=3,  n_layers=1, drop_prob=0.0, input_action_dim=3,
                    input_states_dim=32, output_dim=32)
name_rnn = "GRU_seqin16_seqout3_1layer_45epochs_60000steps_MultiStep2_Soutenance_SCALE13E-1_batch128_Adam_lr5E-3VAL17"
state_dict_gru = torch.load("../trained_rnns/soutenance/" + name_rnn + ".pt")['state_dict']
rnn.load_state_dict(state_dict_gru)

if EVAL_RNN:
    gru.val_gru(rnn, encoder, eval_IRdata_loaded)

random_indexes = [randint(0, len(eval_IRdata_loaded.dataset)) for i in range(NB_SAMPLE)]
theta = np.linspace(0, -256/360*2*np.pi, encoder.dim_input_states)

with torch.no_grad():
    hidden_state = rnn.init_hidden_state(1)

    for batch_index, data in enumerate(eval_IRdata_loaded):
        if batch_index in random_indexes:
            print('sample'+str(batch_index))
            states_in = data['seq_in']['states']
            states_VAE = torch.Tensor(1, rnn.seq_in, rnn.input_states_dim)
            for index_dim in range(rnn.seq_in):
              moy, logvar = encoder.encode(states_in[:, index_dim])
              states_VAE[0, index_dim] = encoder.reparametrize(moy, logvar)

            states_out = data['seq_out']['states']
            target = torch.Tensor(1, rnn.seq_out, rnn.output_dim)
            for index_dim in range(rnn.seq_out):
                moy, logvar = encoder.encode(states_out[:, index_dim])
                target[0, index_dim] = encoder.reparametrize(moy, logvar)

            actions_in = data['seq_in']['actions']
            actions_out = data['seq_out']['actions']
            output, hidden_state = rnn(hidden_state, states_VAE, actions_in, actions_out)

            loss = F.mse_loss(output, target.float(), reduction='sum').item()

            previous_observation = states_in[0, -1]
            observation = states_out[0, 0]
            observation_VAE = encoder.decode(target[0, 0]).detach()[0]
            prediction_RNN = encoder.decode(output[0, 0]).detach()[0]

            plt.plot(states_VAE[0, -1], color='lightgray', label='$z_{t}$')
            plt.plot(target[0, 0], color='fuchsia', label='$z_{t+1}$')
            plt.plot(output[0, 0], color='darkred', label='$z^{*}_{t+1}$')
            loss_pred = F.mse_loss(target[0, 0], output[0, 0],reduction='sum').item()
            plt.title('$L_2(z_{t+1},z^{*}_{t+1})='+str(round(loss_pred,2))+'$', fontsize=20)
            plt.legend(markerscale=15, fontsize=20, handletextpad=0.1)
            plt.show()

            X_old = previous_observation*np.cos(theta)
            Y_old = previous_observation*np.sin(theta)
            X_true = observation * np.cos(theta)
            Y_true = observation * np.sin(theta)
            X_recons = np.array(observation_VAE) * np.cos(theta)
            Y_recons = np.array(observation_VAE) * np.sin(theta)
            X_pred = np.array(prediction_RNN) * np.cos(theta)
            Y_pred = np.array(prediction_RNN) * np.sin(theta)


            plt.scatter(X_old, Y_old, s=0.3, color='black', marker = 'o', label='$o_{t}$')
            plt.scatter(X_true, Y_true, s=0.3, color='darkgreen', marker='o', label='$o_{t+1}$')
            plt.scatter(X_recons, Y_recons, s=0.3, color='fuchsia', marker='x', label = '$\hat{o}_{t+1}$')
            plt.scatter(X_pred, Y_recons, s=0.3, color='red', marker='x', label='$\hat{o}^{*}_{t+1}$')
            plt.scatter(0, 0, s=2, color='red')
            plt.axis('equal')
            loss_recons_reconspred = F.mse_loss(observation_VAE, prediction_RNN,reduction='sum').item()
            loss_obs_reconspred = F.mse_loss(observation, prediction_RNN,reduction='sum').item()
            plt.title('$L_2(ô_{t+1},ô^{*}_{t+1})='
                      + str(round(loss_recons_reconspred, 2))
                      + '\ \ \ \ \ \ L_2(o_{t+1},ô^{*}_{t+1})='
                      + str(round(loss_obs_reconspred, 2)) + '$', fontsize=20)
            plt.legend(markerscale=15, fontsize=20, handletextpad=0.1)
            plt.show()

            if rnn.seq_out==3:
                observation_VAE3 = encoder.decode(target[0, 2]).detach()[0]
                prediction_RNN3 = encoder.decode(output[0, 2]).detach()[0]

                X_recons3 = np.array(observation_VAE3) * np.cos(theta)
                Y_recons3 = np.array(observation_VAE3) * np.sin(theta)
                X_pred3 = np.array(prediction_RNN3) * np.cos(theta)
                Y_pred3 = np.array(prediction_RNN3) * np.sin(theta)

                plt.scatter(X_recons, Y_recons, s=0.1, marker='o', color='red', label='$\hat{o}_{t+1}$')
                plt.scatter(X_pred, Y_pred, s=0.1, marker='x', color='orange', label='$\hat{o}^{*}_{t+1}$')
                plt.scatter(X_recons3, Y_recons3, s=0.1, marker='o', color='royalblue', label='$\hat{o}_{t+3}$')
                plt.scatter(X_pred3, Y_pred3, s=0.1, marker='x', color='limegreen', label='$\hat{o}^{*}_{t+3}$')
                plt.legend(markerscale=20, fontsize=20, handletextpad=0.1)
                loss_recons_reconspred3 = F.mse_loss(observation_VAE3, prediction_RNN3,reduction='sum').item()
                plt.title('$L_2(ô_{t+1},ô^{*}_{t+1})='
                          + str(round(loss_recons_reconspred, 2))
                          + '\ \ \ \ \ \ L_2(ô_{t+3},ô^{*}_{t+3})='
                          + str(round(loss_recons_reconspred3, 2)) + '$', fontsize=20)
                plt.show()

            if rnn.seq_out==5:
                observation_VAE5 = encoder.decode(target[0, 4]).detach()[0]
                prediction_RNN5 = encoder.decode(output[0, 4]).detach()[0]

                X_recons5 = np.array(observation_VAE5) * np.cos(theta)
                Y_recons5 = np.array(observation_VAE5) * np.sin(theta)
                X_pred5 = np.array(prediction_RNN5) * np.cos(theta)
                Y_pred5 = np.array(prediction_RNN5) * np.sin(theta)

                plt.scatter(X_recons, Y_recons, s=0.1, marker='o', color='red', label='$\hat{o}_{t+1}$')
                plt.scatter(X_pred, Y_pred, s=0.1, marker='x', color='orange', label='$\hat{o}^{*}_{t+1}$')
                plt.scatter(X_recons5, Y_recons5, s=0.1, marker='o', color='royalblue', label='$\hat{o}_{t+5}$')
                plt.scatter(X_pred5, Y_pred5, s=0.1, marker='x', color='limegreen', label='$\hat{o}^{*}_{t+5}$')
                plt.legend(markerscale=20, fontsize=20, handletextpad=0.1)
                loss_recons_reconspred5 = F.mse_loss(observation_VAE5, prediction_RNN5, reduction='sum').item()
                plt.title('$L_2(ô_{t+1},ô^{*}_{t+1})='
                          + str(round(loss_recons_reconspred, 2))
                          + '\ \ \ \ \ \ L_2(ô_{t+5},ô^{*}_{t+5})='
                          + str(round(loss_recons_reconspred5, 2)) + '$', fontsize=20)
                plt.show()
