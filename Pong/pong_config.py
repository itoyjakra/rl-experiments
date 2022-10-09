from dataclasses import dataclass

frame_size = 80


@dataclass
class ConvLayer:
    kernel_size: int
    stride: int
    channels: int
    padding: int = 0
    dilation: int = 1

    def out_size(self, in_size: int) -> int:
        return int(
            (in_size + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )


layer1 = ConvLayer(kernel_size=8, stride=4, channels=32)
layer2 = ConvLayer(kernel_size=4, stride=2, channels=64)
layer3 = ConvLayer(kernel_size=3, stride=1, channels=64)

output_size_conv1 = layer1.out_size(in_size=frame_size)
output_size_conv2 = layer2.out_size(in_size=output_size_conv1)
output_size_conv3 = layer3.out_size(in_size=output_size_conv2)
input_size_fc1 = layer3.channels * output_size_conv3 * output_size_conv3
