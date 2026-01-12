class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False, alpha=None, beta=None):
        super(ResUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_res = is_res

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.alpha = alpha if alpha is not None else nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1), requires_grad=True)
        self.beta = beta if beta is not None else nn.Parameter(torch.empty(1).uniform_(0.9, 1.1), requires_grad=True)


    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.is_res:
            out = self.alpha * out + self.beta * x
            # out =  out +  x
        return out


class PAC_ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_units=3, is_res=False, in_POI=None, in_ext=None, grid_size=(18, 22)):
        super().__init__()
        self.grid_size = grid_size
        self.w_img = nn.Parameter(torch.ones((1, *grid_size)), requires_grad=True)

        self.convInput = nn.Conv2d(in_channels, 32, 1, padding="same")
        self.convOutput = nn.Conv2d(32, out_channels, 1, padding="same")

        self.ResNet = nn.Sequential(*[ResUnit(32, 32, is_res) for _ in range(num_res_units)])

        self.POIUnit = nn.Sequential(nn.BatchNorm2d(in_POI), nn.Conv2d(in_POI, 1, 3, padding="same")) if in_POI else None
        self.extUnit = nn.Sequential(nn.Conv2d(in_ext, 1, 1, padding="same")) if in_ext else None

        if in_POI:
            self.w_POI = nn.Parameter(torch.zeros((1, *grid_size)), requires_grad=True)
        if in_ext:
            self.w_ext = nn.Parameter(torch.zeros((1, *grid_size)), requires_grad=True)

    def forward(self, x, POI=None, ext=None):
        out = self.convOutput(self.ResNet(self.convInput(x)))
        res = self.w_img * out
        if POI is not None:
            res += self.w_POI * self.POIUnit(POI)
        if ext is not None:
            res += self.w_ext * self.extUnit(ext)
        return res
