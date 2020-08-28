import torch
import torch.nn as nn
import torch.nn.functional as F

# Des formules utiles pour connaître la taille d'un état en sortie d'une convolution/déconvolution :
# conv
# Lout = int[(Lin + 2*padding - kernel_size)/stride + 1]
# deconv
# Lout = (Lin - 1)*stride - 2*padding + kernel_size + output_padding

# Trois VAE différents :
# • VAEHigh : > Peut encoder des états de dimension 1024 en dimension 384
#             > Peut encoder des états de dimension 512 en dimension 128
# • VAEMediumHigh : > Peut encoder des états de dimension 1024 en dimension 96
#                   > Peut encoder des états de dimension 512 en dimension 32
# • VAEMediumLow :  > Peut encoder des états de dimension 1024 en dimension 128
# • VAELow :  > Peut encoder des états de dimension 1024 en dimension 32


class VAEHigh(nn.Module):
    def __init__(self, dim_input_states=1024):
        super(VAEHigh, self).__init__()

        self.dim_input_states = dim_input_states
        if dim_input_states == 512:
            self.dim_latent_states = (1, 128)  # Calculé directement à partir des formules
        else:
            self.dim_latent_states = (3, 128)  # Calculé directement à partir des formules

        self.conv1 = nn.Conv1d(1,   8,    kernel_size=7, stride=4, padding=0)
        self.conv2 = nn.Conv1d(8,   32,   kernel_size=7, stride=4, padding=0)
        self.conv3 = nn.Conv1d(32,  64,   kernel_size=7, stride=4, padding=0)
        self.conv4 = nn.Conv1d(64,  128,  kernel_size=7, stride=4, padding=0)

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        # self.fc_moy = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        # self.fc_var = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        self.fc_moy = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        self.fc_var = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)

        self.convt1 = nn.ConvTranspose1d(128, 64, 7,  stride=4,   padding=0,  output_padding=0)
        self.convt2 = nn.ConvTranspose1d(64,  32, 7,  stride=4,   padding=0,  output_padding=0)
        self.convt3 = nn.ConvTranspose1d(32,  8,  7,  stride=4,   padding=0,  output_padding=0)
        self.convt4 = nn.ConvTranspose1d(8,   1,  8,  stride=4,   padding=0,  output_padding=0)

    def encode(self, states):
        input_states = states.float()
        input_states = input_states.view(-1, 1, self.dim_input_states)

        # print(input_states.size())
        input_states = self.conv1(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv2(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv3(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv4(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        input_states = input_states.view(-1, dim1_latent*dim2_latent)
        # input_states = input_states.view(-1,1,dim1_latent*dim2_latent)
        # print(input_states.size())

        moy = self.fc_moy(input_states)
        logvar = self.fc_var(input_states)

        return moy, logvar

    def reparametrize(self, moy, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return moy + eps*std  # latent state

    def decode(self, latent_states):
        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        latent_states = latent_states.view(-1, dim2_latent, dim1_latent)

        latent_states = self.convt1(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt2(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt3(latent_states)
        latent_states = F.relu(latent_states)
        # print(z.size())
        latent_states = self.convt4(latent_states)
        latent_states = torch.sigmoid(latent_states)
        # print(latent_states.size())

        output = latent_states.view(-1, self.dim_input_states)

        return output

    def forward(self, states):
        moy, logvar = self.encode(states)
        latent_state = self.reparametrize(moy, logvar)
        return self.decode(latent_state), moy, logvar


class VAEMediumHigh(nn.Module):
    def __init__(self, dim_input_states=1024):
        super(VAEMediumHigh, self).__init__()

        self.dim_input_states = dim_input_states
        if dim_input_states == 512:
            self.dim_latent_states = (1, 32)  # Calculé directement à partir des formules
        else:
            self.dim_latent_states = (3, 32)  # Calculé directement à partir des formules

        self.conv1 = nn.Conv1d(1,   4,    7, stride=4, padding=0)
        self.conv2 = nn.Conv1d(4,   8,   7, stride=4, padding=0)
        self.conv3 = nn.Conv1d(8,  16,   7, stride=4, padding=0)
        self.conv4 = nn.Conv1d(16,  32,  7, stride=4, padding=0)

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        self.fc_moy = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        self.fc_var = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)

        self.convt1 = nn.ConvTranspose1d(32, 16, 7,  stride=4,   padding=0,  output_padding=0)
        self.convt2 = nn.ConvTranspose1d(16,  8, 7,  stride=4,   padding=0,  output_padding=0)
        self.convt3 = nn.ConvTranspose1d(8,  4,  7,  stride=4,   padding=0,  output_padding=0)
        self.convt4 = nn.ConvTranspose1d(4,   1,  8,  stride=4,   padding=0,  output_padding=0)

    def encode(self, states):
        input_states = states.float()

        input_states = input_states.view(-1, 1, self.dim_input_states)
        # print(input_states.size())
        input_states = self.conv1(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv2(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv3(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv4(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        input_states = input_states.view(-1, dim1_latent*dim2_latent)
        # input_states = input_states.view(-1,1,dim1_latent*dim2_latent)
        # print(input_states.size())
        moy = self.fc_moy(input_states)
        logvar = self.fc_var(input_states)

        return moy, logvar

    def reparametrize(self, moy, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return moy + eps*std  # latent state

    def decode(self, latent_states):

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        latent_states = latent_states.view(-1, dim2_latent, dim1_latent)
        latent_states = self.convt1(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt2(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt3(latent_states)
        latent_states = F.relu(latent_states)
        # print(z.size())
        latent_states = self.convt4(latent_states)
        latent_states = torch.sigmoid(latent_states)
        # print(latent_states.size())
        output = latent_states.view(-1, self.dim_input_states)

        return output

    def forward(self, states):
        moy, logvar = self.encode(states)
        latent_state = self.reparametrize(moy, logvar)
        return self.decode(latent_state), moy, logvar


class VAEMediumLow(nn.Module):
    def __init__(self):
        super(VAEMediumLow, self).__init__()

        self.dim_input_states = 1024
        self.dim_latent_states = (1, 128)

        self.conv1 = nn.Conv1d(1,   8,    11, stride=4, padding=0)
        self.conv2 = nn.Conv1d(8,   32,   11, stride=4, padding=0)
        self.conv3 = nn.Conv1d(32,  64,   11, stride=4, padding=0)
        self.conv4 = nn.Conv1d(64,  128,  11, stride=4, padding=0)

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        self.fc_moy = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        self.fc_var = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)

        self.convt1 = nn.ConvTranspose1d(128, 64, 13,  stride=4,   padding=0,  output_padding=0)
        self.convt2 = nn.ConvTranspose1d(64,  32, 13,  stride=4,   padding=0,  output_padding=0)
        self.convt3 = nn.ConvTranspose1d(32,  8,  13,  stride=4,   padding=0,  output_padding=0)
        self.convt4 = nn.ConvTranspose1d(8,   1,  16,  stride=4,   padding=0,  output_padding=0)

    def encode(self, states):
        input_states = states.float()

        input_states = input_states.view(-1, 1, self.dim_input_states)
        # print(input_states.size())
        input_states = self.conv1(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv2(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv3(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv4(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        input_states = input_states.view(-1, dim1_latent*dim2_latent)
        # input_states = input_states.view(-1,1,dim1_latent*dim2_latent)
        # print(input_states.size())
        moy = self.fc_moy(input_states)
        logvar = self.fc_var(input_states)

        return moy, logvar

    def reparametrize(self, moy, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return moy + eps*std  # latent state

    def decode(self, latent_states):

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        latent_states = latent_states.view(-1, dim2_latent, dim1_latent)
        latent_states = self.convt1(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt2(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt3(latent_states)
        latent_states = F.relu(latent_states)
        # print(z.size())
        latent_states = self.convt4(latent_states)
        latent_states = torch.sigmoid(latent_states)
        # print(latent_states.size())
        output = latent_states.view(-1, self.dim_input_states)

        return output

    def forward(self, states):
        moy, logvar = self.encode(states)
        latent_state = self.reparametrize(moy, logvar)
        return self.decode(latent_state), moy, logvar


class VAELow(nn.Module):
    def __init__(self):
        super(VAELow, self).__init__()

        self.dim_input_states = 1024
        self.dim_latent_states = (1, 32)

        self.conv1 = nn.Conv1d(1,   4,    11, stride=4, padding=0)
        self.conv2 = nn.Conv1d(4,   8,   11, stride=4, padding=0)
        self.conv3 = nn.Conv1d(8,  16,   11, stride=4, padding=0)
        self.conv4 = nn.Conv1d(16,  32,  11, stride=4, padding=0)

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        self.fc_moy = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)
        self.fc_var = nn.Linear(dim1_latent*dim2_latent, dim1_latent*dim2_latent)

        self.convt1 = nn.ConvTranspose1d(32, 16, 13,  stride=4,   padding=0,  output_padding=0)
        self.convt2 = nn.ConvTranspose1d(16,  8, 13,  stride=4,   padding=0,  output_padding=0)
        self.convt3 = nn.ConvTranspose1d(8,  4,  13,  stride=4,   padding=0,  output_padding=0)
        self.convt4 = nn.ConvTranspose1d(4,   1,  16,  stride=4,   padding=0,  output_padding=0)

    def encode(self, states):
        input_states = states.float()

        input_states = input_states.view(-1, 1, self.dim_input_states)
        # print(input_states.size())
        input_states = self.conv1(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv2(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv3(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())
        input_states = self.conv4(input_states)
        input_states = F.relu(input_states)
        # print(input_states.size())

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        input_states = input_states.view(-1, dim1_latent*dim2_latent)
        # input_states = input_states.view(-1,1,dim1_latent*dim2_latent)
        # print(input_states.size())
        moy = self.fc_moy(input_states)
        logvar = self.fc_var(input_states)

        return moy, logvar

    def reparametrize(self,moy,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return moy + eps*std  # latent state

    def decode(self, latent_states):

        dim1_latent = self.dim_latent_states[0]
        dim2_latent = self.dim_latent_states[1]
        latent_states = latent_states.view(-1, dim2_latent, dim1_latent)
        latent_states = self.convt1(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt2(latent_states)
        latent_states = F.relu(latent_states)
        # print(latent_states.size())
        latent_states = self.convt3(latent_states)
        latent_states = F.relu(latent_states)
        # print(z.size())
        latent_states = self.convt4(latent_states)
        latent_states = torch.sigmoid(latent_states)
        # print(latent_states.size())
        output = latent_states.view(-1, self.dim_input_states)

        return output

    def forward(self, states):
        moy, logvar = self.encode(states)
        latent_state = self.reparametrize(moy, logvar)
        return self.decode(latent_state), moy, logvar


def loss_vae(recons_states, states, moy, logvar):
    # KLD = -0.5 * torch.sum(1 + logvar - moy.pow(2) - logvar.exp())
    L2 = F.mse_loss(recons_states, states, reduction='sum')
    # return L2 + KLD
    return L2


losses_training = []
train_loss = 1


def train_vae(VAE, optimizer, epoch, train_loaded, agent_name, sensor_name):
    global losses_training
    global train_loss

    VAE.train()
    nb_batches = len(train_loaded)
    batch_size = train_loaded.batch_size
    nb_sample = nb_batches*batch_size

    for batch_index, data in enumerate(train_loaded):
        optimizer.zero_grad()

        target = data['states'][agent_name][sensor_name]
        output, moy, logvar = VAE(data['states'][agent_name][sensor_name])
        loss = loss_vae(output.float(), target.float(), moy, logvar)

        if batch_index == 0 and epoch == 1:
            train_loss = loss.item()/batch_size
        else:
            train_loss = train_loss + 0.01*(loss.item()/batch_size-train_loss)
        losses_training.append(train_loss)

        loss.backward()
        optimizer.step()
        output = output.detach()

        if batch_index % (nb_batches//100+1) == 0:
            print('Train: Epoch {} batch {}/{} ({:.0f}%)\ttraining loss: {:.6f}'.format(
                epoch, batch_index * batch_size, nb_sample, 100. * batch_index / nb_batches, train_loss))

    return losses_training


def val_vae(VAE, eval_loaded, agent_name, sensor_name):
    VAE.eval()
    val_loss = 0
    nb_batches = len(eval_loaded)
    batch_size = eval_loaded.batch_size
    nb_sample = nb_batches*batch_size

    with torch.no_grad():
        for batch_index, data in enumerate(eval_loaded):
            target = data['states'][agent_name][sensor_name]
            output, moy, var = VAE(data['states'][agent_name][sensor_name])
            val_loss += loss_vae(output.float(), target.float(), moy, var).item()

            if batch_index % (nb_batches//100+1) == 0:
                print('computing evaluation loss : [{}/{} ({:.0f}%)]\t'.format(
                  batch_index*batch_size, nb_sample, 100. * batch_index / nb_batches))

    print('evaluation loss : {}'.format(val_loss / nb_sample))

