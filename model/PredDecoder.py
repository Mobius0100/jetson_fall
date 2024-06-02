import torch.nn as nn

class CNN_layer(nn.Module):
    # This is the simple CNN layer,that performs a 2-D convolution
    # while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
        (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            , nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


class PredDecoder(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, T_in, C, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels=2,
                 output_channels=2,
                 input_time_frame=10,
                 output_time_frame=25,
                 joints_to_consider=17,
                 n_txcnn_layers=4,
                 txc_kernel_size=[3,3],
                 txc_dropout=0,
                 bias=True):

        super(PredDecoder, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.gcn = nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0)
        # self.gcn = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0),
        #                          nn.ConvTranspose2d(output_channels, output_channels, kernel_size=1, stride=2,padding=0))
        #(N, T, C, V) -->
        self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size,
                                     txc_dropout))  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        # input: N,C,T,V
        x = self.gcn(x).permute(0, 2, 1, 3)

        x = self.prelus[0](self.txcnns[0](x))
        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x
