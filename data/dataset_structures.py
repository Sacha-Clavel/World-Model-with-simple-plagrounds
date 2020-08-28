import numpy as np
import pickle
from torch.utils.data import Dataset

# Pour charger les données collectées au cours d'une simulation d'une durée de T pas de temps.
# Permet d'avoir une structure de données réutilisable pour entraîner divers réseaux de neuronnes.
# Les données stockées correspondent aux observations courrantes et suivantes de chaque capteur de chaque agent,
# et aux actions appliquées à chaque partie de chaque agent.
# Les données sont stockées dans un dictionnaires, comme dans cet exemple :

# data = {
#   'states' : {
#       'agent1': {
#           'sensor_IR' : [o_0, o_1, ..., o_T-2]
#           'sensor_RGB' :  [o_0, o_1, ..., o_T-2]
#           }
#       'agent2' : {
#           'sensor_IR' : [o_0, o_1, ..., o_T-2]
#           'sensor_RGB' : [o_0, o_1, ..., o_T-2]
#           }
#       }
#   'actions' : {
#       'agent1':{
#           'base' : {
#               'LONGITUDINAL_VELOCITY' : [a_0, a_1, ..., a_T-2]
#               'ANGULAR_VELOCITY' : [a_0, a_1, ..., a_T-2]
#               }
#           'arm' : {
#               'ANGULAR_VELOCITY' : [a_0, a_1, ..., a_T-2]
#               'GRASP' : [a_0, a_1, ..., a_T-2]
#               }
#           }
#       }
#   'next_states' : {
#       'agent1': {
#             'sensor_IR' : [o_1, o_2, ..., o_T-1]
#             'sensor_RGB' :  [o_1, o_2, ..., o_T-1]
#             }
#       'agent2' : {
#             'sensor_IR' : [o_1, o_2, ..., o_T-1]
#             'sensor_RGB' : [o_1, o_2, ..., o_T-1]
#             }
#         }
# }

# En pratique j'ai utilisé cette structure de données uniquement pour entraîner le VAE.
# J'ai trouvé utile de la conserver car elle peut être réutilisée


class DataRecord(Dataset):
    def __init__(self, p_file, train, train_rate=0.8, step_sample=1, step_dot=1):

        print("Loading rawdata ...")
        fid = open(p_file, "rb")
        rawdata = pickle.load(fid)

        self.agent_names = list(rawdata['observations'].keys())
        self.minimaxis = {'states': {}, 'actions': {}}  # Stockage des minimas/maximas des observations/actions

        print("Normalizing rawdata ...")
        for agent_key in self.agent_names:
            self.minimaxis['states'][agent_key] = {}
            for sensor_key in rawdata['observations'][agent_key].keys():  # Normalisation des observations : observation = (observation-min)/(max-min)
                rawdata['observations'][agent_key][sensor_key] = np.array(
                    rawdata['observations'][agent_key][sensor_key])
                max_observation = np.max(rawdata['observations'][agent_key][sensor_key])
                min_observation = np.min(rawdata['observations'][agent_key][sensor_key])
                self.minimaxis['states'][agent_key][sensor_key] = (min_observation, max_observation)
                rawdata['observations'][agent_key][sensor_key] = (rawdata['observations'][agent_key][sensor_key]
                                                                  - min_observation) / (
                                                                             max_observation - min_observation)

            self.minimaxis['actions'][agent_key] = {}
            for body_part_key in rawdata['actions'][agent_key].keys():
                self.minimaxis['actions'][agent_key][body_part_key] = {}
                for action_key in rawdata['actions'][agent_key][body_part_key].keys():
                    rawdata['actions'][agent_key][body_part_key][action_key] = np.array(
                        rawdata['actions'][agent_key][body_part_key][action_key])

                    max_action = np.max(rawdata['actions'][agent_key][body_part_key][action_key])
                    min_action = np.min(rawdata['actions'][agent_key][body_part_key][action_key])
                    self.minimaxis['actions'][agent_key][body_part_key][action_key] = (min_action, max_action)
                    # Normalisation des actions inutile car les actions sont comprises entre -1 et 1.
                    # rawdata['actions'][agent_key][body_part_key][action_key] = (rawdata['actions'][agent_key][body_part_key][action_key]
                    #                                                             - min_action)/(max_action-min_action)

        sensor_names_agent0 = list(rawdata['observations'][self.agent_names[0]].keys())
        train_index = int(len(rawdata['observations'][self.agent_names[0]][sensor_names_agent0[0]]) * train_rate)
        # Pour séparer les données : Données = training_set + testing_set = [0, ..., train_index[ + [train_index, ..., T-1]

        self.data = {'states': {},
                     'actions': {},
                     'next_states': {}}
        if train:
            print("Storing data for training ...")
            for agent_key in self.agent_names:
                self.data['states'][agent_key] = {}
                self.data['next_states'][agent_key] = {}
                self.data['actions'][agent_key] = {}

                for sensor_key in rawdata['observations'][agent_key].keys():
                    if 'IR' in sensor_key:
                        self.data['states'][agent_key][sensor_key] = rawdata['observations'][agent_key][sensor_key][
                                                                     0:train_index - 1:step_sample,
                                                                     28:1052:step_dot]  # 28:1052 pour avoir 1024 points
                        self.data['next_states'][agent_key][sensor_key] = rawdata['observations'][agent_key][
                                                                              sensor_key][1:train_index:step_sample,
                                                                          28:1052:step_dot]
                    else:
                        self.data['states'][agent_key][sensor_key] = rawdata['observations'][agent_key][sensor_key][
                                                                     0:train_index - 1:step_sample, 0::step_dot]
                        self.data['next_states'][agent_key][sensor_key] = rawdata['observations'][agent_key][
                                                                              sensor_key][1:train_index:step_sample,
                                                                                0::step_dot]

                for body_part_key in rawdata['actions'][agent_key].keys():
                    self.data['actions'][agent_key][body_part_key] = {}
                    for action_key in rawdata['actions'][agent_key][body_part_key].keys():
                        self.data['actions'][agent_key][body_part_key][action_key] = \
                            rawdata['actions'][agent_key][body_part_key][action_key][0:train_index - 1:step_sample]
        else:
            print("Storing data for evaluation ...")
            for agent_key in self.agent_names:
                self.data['states'][agent_key] = {}
                self.data['next_states'][agent_key] = {}
                self.data['actions'][agent_key] = {}

                for sensor_key in rawdata['observations'][agent_key].keys():
                    if 'IR' in sensor_key:
                        self.data['states'][agent_key][sensor_key] = rawdata['observations'][agent_key][sensor_key][
                                                                     train_index:-1:step_sample, 28:1052:step_dot]
                        self.data['next_states'][agent_key][sensor_key] = rawdata['observations'][agent_key][
                                                                              sensor_key][train_index + 1::step_sample,
                                                                          28:1052:step_dot]
                    else:
                        self.data['states'][agent_key][sensor_key] = rawdata['observations'][agent_key][sensor_key][
                                                                     train_index:-1:step_sample, 0::step_dot]
                        self.data['next_states'][agent_key][sensor_key] = rawdata['observations'][agent_key][
                                                                              sensor_key][train_index + 1::step_sample,
                                                                          0::step_dot]

                for body_part_key in rawdata['actions'][agent_key].keys():
                    self.data['actions'][agent_key][body_part_key] = {}
                    for action_key in rawdata['actions'][agent_key][body_part_key].keys():
                        self.data['actions'][agent_key][body_part_key][action_key] = \
                            rawdata['actions'][agent_key][body_part_key][action_key][train_index:-1:step_sample]

    def __len__(self):
        sensor_names_agent0 = list(self.data['states'][self.agent_names[0]].keys())
        return len(self.data['states'][self.agent_names[0]][sensor_names_agent0[0]])

    def __getitem__(self, index):

        sample = {'states': {}, 'actions': {}, 'next_states': {}}

        for agent_key in self.agent_names:
            sample['states'][agent_key] = {}
            sample['actions'][agent_key] = {}
            sample['next_states'][agent_key] = {}
            for sensor_key in self.data['states'][agent_key].keys():
                sample['states'][agent_key][sensor_key] = self.data['states'][agent_key][sensor_key][index]
                sample['next_states'][agent_key][sensor_key] = self.data['next_states'][agent_key][sensor_key][index]

            for body_part_key in self.data['actions'][agent_key].keys():
                sample['actions'][agent_key][body_part_key] = {}
                for action_key in self.data['actions'][agent_key][body_part_key].keys():
                    sample['actions'][agent_key][body_part_key][action_key] = \
                        self.data['actions'][agent_key][body_part_key][action_key][index]

        return sample


# Pour grouper les observations et actions en séquences.
# Permet d'entraîner des Recurrent Neural Networks par exemple
# Les séquences sont des dictionnaires et sont construites ainsi :

# séquence du temps t = { seq_in :  {
#                               states :    [ o_t, o_(t+1), ..., o_(t+seq_in-1)]
#                               actions :   [ a_t, a_(t+1), ..., a_(t+seq_in-1)]
#                               }
#                         seq_out : {
#                               states :    [ o_(t+seq_in), o_(t+seq_in+1), ..., o_(t+seq_in+seq_out-1) ]
#                               actions :   [ a_(t+seq_in), a_(t+seq_in+1), ..., a_(t+seq_in+seq_out-1) ]
#                               }
#                         }

class IRRecordSequences(Dataset):
    def __init__(self, p_file, min_observation_IR, max_observation_IR, agent_name='robot', IR_sensor_name='IR_1',
                 train=True, train_rate=0.8, step_dot=1, seq_in=10, seq_out=4):

        print("Loading rawdata ...")
        fid = open(p_file, "rb")
        rawdata = pickle.load(fid)

        train_index = int(len(rawdata['observations'][agent_name][IR_sensor_name]) * train_rate)

        rawdata['observations'][agent_name][IR_sensor_name] = np.array(
            rawdata['observations'][agent_name][IR_sensor_name])
        print("Normalizing rawdata ...")
        rawdata['observations'][agent_name][IR_sensor_name] = (rawdata['observations'][agent_name][IR_sensor_name]
                                                               - min_observation_IR) / (
                                                                      max_observation_IR - min_observation_IR)

        self.seq_in = seq_in
        self.seq_out = seq_out
        self.seq = seq_in + seq_out

        all_actions = []
        for bodypart_key in rawdata['actions'][agent_name].keys():
            for action_key in rawdata['actions'][agent_name][bodypart_key].keys():
                all_actions.append(rawdata['actions'][agent_name][bodypart_key][action_key])
        all_actions = np.array(all_actions)
        all_actions = np.transpose(all_actions)
        rawdata['actions'] = all_actions

        self.data = {}

        if train:
            print("Storing data for training")
            self.data['states'] = rawdata['observations'][agent_name][IR_sensor_name][0:train_index - 1,
                                  28:1052:step_dot]
            self.data['actions'] = rawdata['actions'][1:train_index]

        else:
            print("Storing data for evaluation")
            self.data['states'] = rawdata['observations'][agent_name][IR_sensor_name][train_index:-1, 28:1052:step_dot]
            self.data['actions'] = rawdata['actions'][train_index + 1:]

    def __len__(self):
        return len(self.data['states']) - self.seq

    def __getitem__(self, index):

        sample = {'seq_in': {}, 'seq_out': {}}

        sample['seq_in']['states'] = self.data['states'][index:index + self.seq_in]
        sample['seq_in']['actions'] = self.data['actions'][index:index + self.seq_in]
        sample['seq_out']['states'] = self.data['states'][index + self.seq_in:index + self.seq_in + self.seq_out]
        sample['seq_out']['actions'] = self.data['actions'][index + self.seq_in:index + self.seq_in + self.seq_out]
        return sample
