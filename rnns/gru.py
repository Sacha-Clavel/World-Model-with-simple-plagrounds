import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUNet(nn.Module):
    def __init__(self, seq_in=10, seq_out=1, input_states_dim=32, input_action_dim=4, inner_action_dim=4,
                 hidden_dim=124, output_dim=32, n_layers=1, drop_prob=0.0):
        super(GRUNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.seq_in = seq_in
        self.seq_out = seq_out

        self.input_states_dim = input_states_dim
        self.input_action_dim = input_action_dim
        self.input_dim = input_states_dim + input_action_dim
        self.output_dim = output_dim
        # self.fc_a = nn.Linear(input_action_dim, inner_action_dim)

        self.gru_in = nn.GRU(self.input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # self.gru_in = nn.GRU(self.input_dim+inner_action_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.gru_out = nn.GRU(input_action_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, states_VAE, actions_in, actions_out):
        actions_in = actions_in.float()
        state_vae = states_VAE.float()
        # actions_in = self.fc_a(actions)
        # actions = F.relu(actions)
        x = torch.cat((state_vae, actions_in), 2)

        out, inner_state = self.gru_in(x, h)

        actions_out = actions_out.float()
        prediction, h = self.gru_out(actions_out, inner_state)

        prediction = self.fc(F.relu(prediction))
        return prediction, h

    def init_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


losses_training = []
train_loss = 1


def train_gru(GRU, encoder, optimizer, epoch, train_loaded):
    global losses_training
    global train_loss

    GRU.train()
    nb_batches = len(train_loaded)
    batch_size = train_loaded.batch_size
    nb_sample = len(train_loaded.dataset)

    hidden_state = GRU.init_hidden_state(batch_size)

    for batch_index, data in enumerate(train_loaded):

        with torch.no_grad():
            states_in = data['seq_in']['states']
            states_vae = torch.Tensor(batch_size, GRU.seq_in, GRU.input_states_dim)
            for index_dim in range(GRU.seq_in):
                moy, logvar = encoder.encode(states_in[:, index_dim])
                states_vae[:, index_dim] = encoder.reparametrize(moy, logvar)

            states_out = data['seq_out']['states']
            target = torch.Tensor(batch_size, GRU.seq_out, GRU.output_dim)
            for index_dim in range(GRU.seq_out):
                moy, logvar = encoder.encode(states_out[:, index_dim])
                target[:, index_dim] = encoder.reparametrize(moy, logvar)

        optimizer.zero_grad()
        actions_in = data['seq_in']['actions']
        actions_out = data['seq_out']['actions']

        output, hidden_state = GRU(hidden_state, states_vae, actions_in, actions_out)

        loss = F.mse_loss(target.float(), output, reduction='sum')

        if batch_index == 0 and epoch == 1:
            train_loss = loss.item()/batch_size
        else:
            train_loss = train_loss + 0.01*(loss.item()/batch_size-train_loss)
        losses_training.append(train_loss)

        loss.backward()
        optimizer.step()
        output = output.detach()
        hidden_state = hidden_state.detach()

        if batch_index % (nb_batches // 100 + 1) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_loss: {:.6f}'.format(
                epoch, batch_index * batch_size, nb_sample,
                100. * batch_index / nb_batches, train_loss))


def val_gru(GRU, encoder, eval_loaded):
    GRU.eval()
    val_loss = 0
    nb_batches = len(eval_loaded)
    batch_size = eval_loaded.batch_size
    nb_sample = len(eval_loaded.dataset)

    hidden_state = GRU.init_hidden_state(batch_size)

    with torch.no_grad():
        for batch_index, data in enumerate(eval_loaded):
            states_in = data['seq_in']['states']
            states_VAE = torch.Tensor(batch_size, GRU.seq_in, GRU.input_states_dim)
            for index_dim in range(GRU.seq_in):
                moy, logvar = encoder.encode(states_in[:, index_dim])
                states_VAE[:, index_dim] = encoder.reparametrize(moy, logvar)

            states_out = data['seq_out']['states']
            target = torch.Tensor(batch_size, GRU.seq_out, GRU.output_dim)
            for index_dim in range(GRU.seq_out):
                moy, logvar = encoder.encode(states_out[:, index_dim])
                target[:, index_dim] = encoder.reparametrize(moy, logvar)

            actions_in = data['seq_in']['actions']
            actions_out = data['seq_out']['actions']
            output, hidden_state = GRU(hidden_state, states_VAE, actions_in, actions_out)

            val_loss += F.mse_loss(output, target.float(), reduction='sum').item()

            if batch_index % (nb_batches // 100 + 1) == 0:
                print('computing evaluation loss : [{}/{} ({:.0f}%)]\t'.format(
                    batch_index * batch_size, nb_sample, 100. * batch_index / nb_batches))

    print('evaluation loss : {}'.format(val_loss / nb_sample))
