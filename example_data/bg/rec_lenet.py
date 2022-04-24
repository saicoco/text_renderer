from paddle import nn
import paddle.nn.functional as F
import paddle


__all__ = ["RecTinyNet"]


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        if_act=True,
        act="relu",
    ):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )

        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()
        return x


class RecTinyNet(nn.Layer):
    def __init__(self, in_channels=3, out_channels=256) -> None:
        super(RecTinyNet, self).__init__()
        self.conv1 = ConvBNLayer(
            in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = ConvBNLayer(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = ConvBNLayer(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.max_pool3 = nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1))
        self.conv4 = ConvBNLayer(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1
        )
        self.max_pool4 = nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1))
        
        self.out_channels = out_channels
        self.conv5 = ConvBNLayer(
            in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.max_pool5 = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = self.max_pool4(x)
        x = self.conv5(x)
        x = self.max_pool5(x)
        return x


if __name__ == "__main__":
    import numpy as np

    a = np.ones((1, 3, 32, 320), dtype=np.float32)
    model = RecTinyNet()
    x = paddle.to_tensor(a, dtype="float32")
    out = model(x)
    print(out.shape)
