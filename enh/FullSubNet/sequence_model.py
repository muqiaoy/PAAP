import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        Wrapper of conventional sequence models (LSTM or GRU)

        Args:
            input_size: 每帧输入特征大小
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise, no linear layer.
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "SRU":
            pass
            # self.sequence_model = CustomSRU(
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     num_layers=num_layers,
            #     bidirectional=bidirectional,
            #     highway_bias=-2
            # )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"The shape of input is {x.shape}."
        self.sequence_model.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.permute(0, 2, 1)  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)

        if self.output_size:
            o = self.fc_output_layer(o)

        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
        return o