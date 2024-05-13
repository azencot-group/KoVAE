import torch
from torchdiffeq import odeint

from models.gru_ode_models import FullGRUODECell_Autonomous

class RecoveryODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, last_activation='identity', solver='euler'):
        ''' 24 24 6 24 48
        Arguments:
            input_size: input shape
            hidden_size: shape of hidden state of GRUODE and GRU
            output_size: output shape
            gru_input_size: input size of GRU (raw input will pass through x_model which change shape input_size to gru_input_size)
            x_hidden: shape going through x_model
            delta_t: integration time step for fixed integrator
            solver: ['euler','midpoint','dopri5']
        '''
        super(RecoveryODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)

        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = torch.nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(
            self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            # 관측치가 없을 때 -> 적분
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            # 관측치가 있을 때 -> 점프
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        X_tilde = self.rec_linear(out)
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class First_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super(First_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            # 관측치가 없을 때 -> 적분
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            # 관측치가 있을 때 -> 점프
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Mid_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super(Mid_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            # 관측치가 없을 때 -> 적분
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            # 관측치가 있을 때 -> 점프
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Last_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, last_activation='identity', solver='euler'):
        super(Last_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = torch.nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(
            self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            # 관측치가 없을 때 -> 적분
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            # 관측치가 있을 때 -> 점프
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        X_tilde = self.rec_linear(out)
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class Multi_Layer_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, num_layer, last_activation='identity', solver='euler'):
        super(Multi_Layer_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True
        self.num_layer = num_layer
        self.last_activation= last_activation

        if num_layer == 1:
            self.model = RecoveryODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                            gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
        elif num_layer == 2:
            self.model = torch.nn.ModuleList(
                [
                    First_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
                                     gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver),
                    Last_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
                                    gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
                ]
            )
        else:
            self.model = torch.nn.ModuleList()
            for i in range(num_layer):
                if i == 0:
                    self.model.append(First_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver))
                elif i == num_layer-1:
                    self.model.append(Last_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden,last_activation=self.last_activation, delta_t=delta_t, solver=solver))
                else:
                    self.model.append(Mid_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver))

    def forward(self, H, times):
        if self.num_layer == 1:
            out = self.model(H, times)
        else:
            out = H
            for model in self.model:
                out = model(out, times)
        return out