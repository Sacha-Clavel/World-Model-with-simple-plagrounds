import pickle
import random as rd

# Script pour mélanger toutes les données récoltées récoltées

name_file = "IR_data_fov270_1080dots_60000stepsMultiStep=2_soutenance_scale13E-1.p"
path = "recordings/environment_soutenance/"
fid = open(path + name_file, "rb")
rawdata = pickle.load(fid)
fid.close()

data = rawdata.copy()
rd.shuffle(data['observations']['robot']['IR_1'])

fidShuffle = open(path + "SHUFFLED_" + name_file, "wb")
pickle.dump({'observations': data['observations'], 'actions': data['actions']}, fidShuffle)
fidShuffle.close()
