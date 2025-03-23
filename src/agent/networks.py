import torch.nn as nn
import torch
from torch.distributions import Normal
class MLP(nn.Module):
    def __init__(self, net_config,device):
        '''
        :param net_config: a list of dictionaries where each dictionary specifies the
                           configuration of one layer in the network.
                           Example format:
                           [{'in': 2, 'out': 4, 'drop_out': 0.5, 'activation': 'ReLU'},
                            {'in': 4, 'out': 8, 'drop_out': 0, 'activation': 'Sigmoid'},
                            {'in': 8, 'out': 10, 'drop_out': 0, 'activation': 'None'}]
                           The list can have a customizable number of dictionaries.
        '''
        super(MLP, self).__init__()

        self.net = nn.Sequential()
        self.net_config = net_config
        self.__device = device
        self.build_net()

    def build_net(self):
        # Iterate through the network configuration to build each layer
        # print(f"Using device: {device}")
        device = self.__device
        for layer_id, layer_config in enumerate(self.net_config):
            linear = nn.Linear(layer_config['in'], layer_config['out']).to(device)
            self.net.add_module(f"layer{layer_id}-linear", linear)
            drop_out = nn.Dropout(layer_config['drop_out']).to(device)
            self.net.add_module(f"layer{layer_id}-drop_out", drop_out)
            if layer_config['activation'] != 'None':
                activation = eval('nn.' + layer_config['activation'])().to(device)
                self.net.add_module(f"layer{layer_id}-activation", activation)

    def get_config(self):
        print("Model Parameters:\n")
        for name, param in self.named_parameters():
            print(f"Name: {name}")
            print(f"Shape: {param.shape}")
            #print(f"Values: {param}\n")

    def forward(self, x):
        x = x.to(next(self.net.parameters()).device) 
        return self.net(x)


class Actor(nn.Module):
    '''
        Actor is a neural network module designed to select mutation and crossover parameters
            for a differential evolution algorithm.

        This network consists of:
        - An operator selection network to choose a mutation operator.
        - Networks to predict mutation and crossover parameters (means and standard deviations).

        Attributes:
        - input_dim: The dimension of the input state.
        - n_operator: The number of mutation operators to choose from.
        - n_mutation: The number of mutation parameters to output.
        - n_crossover: The number of crossover parameters to output.
        - output_dim: The total number of outputs, including the selected operator and mutation/crossover parameters.
        - max_sigma: The maximum value for the standard deviation of mutation/crossover parameters.
        - min_sigma: The minimum value for the standard deviation of mutation/crossover parameters.
        '''
    def __init__(self, input_dim, mu_operator, cr_operator, n_mutation, n_crossover, device):
        super(Actor, self).__init__()
        self.output_dim = 2 + n_mutation + n_crossover

        # # Configuration for the mutation operator selection network
        # mutation_operator_net_config = [{'in': input_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 32, 'out': mu_operator, 'drop_out': 0, 'activation': 'None'}]
        #
        # # Configuration for the crossover operator selection network
        # crossover_operator_net_config = [{'in': input_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 32, 'out': cr_operator, 'drop_out': 0, 'activation': 'None'}]
        #
        # # Configuration for the mutation parameters
        # mutation_param_net_config = [{'in': input_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 32, 'out': n_mutation, 'drop_out': 0, 'activation': 'None'}]
        #
        # # Configuration for the crossover parameters
        # crossover_param_net_config = [{'in': input_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
        #                                 {'in': 32, 'out': n_crossover, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the mutation operator selection network
        mutation_operator_net_config = [
                                        {'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                            {'in': 32, 'out': mu_operator, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the crossover operator selection network
        crossover_operator_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                            {'in': 32, 'out': cr_operator, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the mutation parameters
        mutation_param_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                        {'in': 32, 'out': n_mutation, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the crossover parameters
        crossover_param_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                        {'in': 32, 'out': n_crossover, 'drop_out': 0, 'activation': 'None'}]

        # Initialize networks for operator selection and parameters
        self.mutation_selector_net = MLP(mutation_operator_net_config,device)
        self.crossover_selector_net = MLP(crossover_operator_net_config,device)

        self.mutation_param_sigma_net = MLP(mutation_param_net_config,device)
        self.mutation_param_mu_net = MLP(mutation_param_net_config,device)
        self.crossover_param_sigma_net = MLP(crossover_param_net_config,device)
        self.crossover_param_mu_net = MLP(crossover_param_net_config,device)

        # Define maximum and minimum sigma values for parameter scaling
        self.max_sigma = 0.7
        self.min_sigma = 0.1

        self.n_mutation = n_mutation
        self.n_crossover = n_crossover

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_action(self, x):
        mutation_p = self.mutation_selector_net(x)  # 2D : ps * mu_operator
        crossover_p = self.crossover_selector_net(x)  # 2D : ps * cr_operator
        # Apply softmax to get probabilities for operator selection

        mutation_p = torch.softmax(mutation_p, dim = 1)
        crossover_p = torch.softmax(crossover_p, dim = 1)

        # Calculate mutation parameters using tanh activation and normalization
        m_mu = (torch.tanh(self.mutation_param_mu_net(x)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma
        # Calculate crossover parameters using tanh activation and normalization
        c_mu = (torch.tanh(self.crossover_param_mu_net(x)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma
        # Step 1: Get indices of maximum probabilities
        mutation_idx = torch.argmax(mutation_p, dim = 1, keepdim = True)  # ps * 1
        crossover_idx = torch.argmax(crossover_p, dim = 1, keepdim = True)  # ps * 1

        # Step 2: Flatten m_mu and c_mu for concatenation
        # m_mu: ps * n_mutation
        # c_mu: ps * n_crossover
        m_mu_flattened = m_mu  # ps * n_mutation
        c_mu_flattened = c_mu  # ps * n_crossover

        # Step 3: Concatenate actions
        # Concatenate mutation_idx, crossover_idx, m_mu_flattened, and c_mu_flattened
        actions = torch.cat([mutation_idx, crossover_idx, m_mu_flattened, c_mu_flattened], dim = 1)  # ps * (2 + n_mutation + n_crossover)

        # Output actions
        return (actions, m_mu, m_sigma, c_mu, c_sigma)

    def get_action_sample(self, x):
        mutation_p = self.mutation_selector_net(x)  # 2D : bs * mu_operator
        crossover_p = self.crossover_selector_net(x)  # 2D : bs * cr_operator
        # Apply softmax to get probabilities for operator selection

        mutation_p = torch.softmax(mutation_p, dim = 1)
        crossover_p = torch.softmax(crossover_p, dim = 1)
        # Create categorical distribution for operator selection
        # try:
        #     # Create categorical distribution for operator selection
        #     mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        # except ValueError as e:
        #     print("Error creating Categorical distribution for mutation_p:", e)
        #     print(x1)
        #     print(x2)
        #     print("mutation_p values:", mutation_p)
        #     raise  # 重新抛出异常，方便定位问题
        mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        crossover_operator_distribution = torch.distributions.Categorical(crossover_p)

        # Calculate mutation parameters using tanh activation and normalization
        m_mu = (torch.tanh(self.mutation_param_mu_net(x)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_net(x)) + 1.) / 2. * (
                self.max_sigma - self.min_sigma) + self.min_sigma

        # Calculate crossover parameters using tanh activation and normalization
        c_mu = (torch.tanh(self.crossover_param_mu_net(x)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_net(x)) + 1.) / 2. * (
                self.max_sigma - self.min_sigma) + self.min_sigma

        # Create Normal distributions for mutation and crossover parameters
        m_policy = Normal(m_mu, m_sigma)
        c_policy = Normal(c_mu, c_sigma)

        mutation_operator_action = mutation_operator_distribution.sample()  # 1D : bs
        crossover_operator_action = crossover_operator_distribution.sample()  # 1D : bs
        mutation_action = torch.clamp(m_policy.sample(), min = 0, max = 1)  # 2D : bs * n_mutation
        crossover_action = torch.clamp(c_policy.sample(), min = 0, max = 1)  # 2D : bs * n_crossover

        # Concatenate all actions into a single tensor
        action = torch.cat([mutation_operator_action[:, None],
                            crossover_operator_action[:, None],
                            mutation_action,
                            crossover_action],
                            dim = 1)  # 2D : bs * (2 + n_mutation + n_crossover)

        return (action, m_mu, m_sigma, c_mu, c_sigma)

    def forward(self, x, fixed_action = None, require_entropy = False):
        mutation_p = self.mutation_selector_net(x)  # 2D : bs * mu_operator
        crossover_p = self.crossover_selector_net(x) # 2D : bs * cr_operator
        # Apply softmax to get probabilities for operator selection

        mutation_p = torch.softmax(mutation_p, dim = 1)
        crossover_p = torch.softmax(crossover_p, dim = 1)
        # Create categorical distribution for operator selection
        # try:
        #     # Create categorical distribution for operator selection
        #     mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        # except ValueError as e:
        #     print("Error creating Categorical distribution for mutation_p:", e)
        #     print(x1)
        #     print(x2)
        #     print("mutation_p values:", mutation_p)
        #     raise  # 重新抛出异常，方便定位问题
        mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        crossover_operator_distribution = torch.distributions.Categorical(crossover_p)

        # Calculate mutation parameters using tanh activation and normalization
        m_mu = (torch.tanh(self.mutation_param_mu_net(x)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma

        # Calculate crossover parameters using tanh activation and normalization
        c_mu = (torch.tanh(self.crossover_param_mu_net(x)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma

        # Create Normal distributions for mutation and crossover parameters
        m_policy = Normal(m_mu, m_sigma)
        c_policy = Normal(c_mu, c_sigma)

        if fixed_action is not None:
            # If fixed actions are provided, use them directly
            mutation_operator_action = fixed_action[:, 0]
            crossover_operator_action = fixed_action[:, 1]
            mutation_action = fixed_action[:, 2: 2 + self.n_mutation]
            crossover_action = fixed_action[:, -self.n_crossover:]
        else:
            # Sample actions from the distributions
            mutation_operator_action = mutation_operator_distribution.sample()  # 1D : bs
            crossover_operator_action = crossover_operator_distribution.sample() # 1D : bs
            mutation_action = torch.clamp(m_policy.sample(), min = 0, max = 1)  # 2D : bs * n_mutation
            crossover_action = torch.clamp(c_policy.sample(), min = 0, max = 1)  # 2D : bs * n_crossover

        # Concatenate all actions into a single tensor
        action = torch.cat([mutation_operator_action[:, None],
                            crossover_operator_action[:, None],
                            mutation_action,
                            crossover_action],
                            dim = 1)  # 2D : bs * (2 + n_mutation + n_crossover)

        # Calculate log probabilities for each action
        mutation_operator_log_prob = mutation_operator_distribution.log_prob(mutation_operator_action)  # 1D : bs
        crossover_operator_log_prob = crossover_operator_distribution.log_prob(crossover_operator_action)  # 1D : bs
        mutation_log_prob = m_policy.log_prob(mutation_action)  # 2D : bs * n_mutation
        crossover_log_prob = c_policy.log_prob(crossover_action)  # 2D : bs * n_crossover

        mutation_log_prob = torch.sum(mutation_log_prob, dim = 1)  # 1D : bs
        crossover_log_prob = torch.sum(crossover_log_prob, dim = 1)  # 1D : bs
        log_prob = mutation_operator_log_prob + \
                    crossover_operator_log_prob + \
                    mutation_log_prob + \
                    crossover_log_prob  # Total log probability
        log_prob = log_prob.sum()
        if require_entropy:
            # Calculate entropy for each distribution if required
            mutation_operator_entropy = mutation_operator_distribution.entropy()
            crossover_operator_entropy = crossover_operator_distribution.entropy()
            mutation_entropy = m_policy.entropy()
            crossover_entropy = c_policy.entropy()
            # Concatenate all entropy values into a single tensor
            entropy = torch.cat([mutation_operator_entropy[:, None],
                                crossover_operator_entropy[:, None],
                                mutation_entropy,
                                crossover_entropy],
                                dim = 1)  # 2D : bs * (2 + n_mutation + n_crossover)
            out = (action, log_prob, entropy)
        else:
            out = (action, log_prob,)

        return out


class Critic(nn.Module):
    def __init__(self, input_dim,device):
        super(Critic, self).__init__()
        net_config = [{'in': input_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 16, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': 1, 'drop_out': 0, 'activation': 'None'}]
        self.__value_head = MLP(net_config,device)
    def forward(self, state):
        baseline_value = self.__value_head(state).mean(1) # [bs, ps, 1] -> [bs, 1]
        return baseline_value.detach(), baseline_value

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

## Test
if __name__ == '__main__':
    policy_net = Actor(16, 14, 3, 2)
    feature = torch.rand(4, 16)
    action, log_prob, entropy = policy_net(feature, require_entropy = True)
    print(policy_net.get_parameter_number())
    policy_net.train()
