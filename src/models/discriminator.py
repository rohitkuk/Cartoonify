import torch
import torch.nn  as nn




class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, activation ,batch_norm ,*args , **kwargs):
        super(convblock, self).__init__()

          
        self.out_channels = out_channels
        self.in_channels = in_channels
        # print(self.conv_architecture)

        self.activations =  nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(0.2)],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()],
            ['Sigmoid', nn.Sigmoid()],

        ])
        self.batch_norms = nn.ModuleDict([
            ['Yes', nn.BatchNorm2d(self.out_channels)],
            ['No', nn.Identity()]

        ])
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding = 1, padding_mode = "reflect", *args, **kwargs, bias = False),
            self.batch_norms[batch_norm],
            self.activations[activation]
        )

    def forward(self, x):
        return self.l1(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.features =  features

        self.initial = convblock(
            in_channels=self.img_channels*2, out_channels = self.features, 
            kernel_size = 4, stride = 2, activation = 'lrelu', batch_norm="No")

        self.layer1 = convblock(
            in_channels = self.features, out_channels = self.features *2 , 
            kernel_size = 4, stride = 2, activation = 'lrelu', batch_norm="Yes")

        self.layer2 = convblock(
            in_channels=self.features *2, out_channels = self.features *4 , 
            kernel_size = 4, stride = 2, activation = 'lrelu', batch_norm="Yes")

        self.layer3 = convblock(
            in_channels=self.features *4, out_channels = self.features *8, 
            kernel_size = 4, stride = 1, activation = 'lrelu', batch_norm="Yes")

        self.layer4 = convblock(
            in_channels=self.features*8, out_channels = 1 , 
            kernel_size = 4, stride = 1, activation = 'Sigmoid', batch_norm="Yes")


    def forward(self, x, y):
        x = self.initial(torch.cat([x, y], 1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(img_channels=3, features = 64)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()