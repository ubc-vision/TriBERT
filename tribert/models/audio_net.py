import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        ### sampler
        #import ipdb; ipdb.set_trace()
        psi = F.upsample(psi, size=x.size()[2:], mode="bilinear", align_corners=True)

        return x*psi


class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(Unet, self).__init__()
        # construct unet structure
        in_channels = 1
        out_channels = 1 #1  #Himu
        init_features = 64
        features = init_features
        self.encoder1 = Unet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Unet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Unet._block(features * 8, features * 24, name="bottleneck")  #Himu features * 16

        self.upconv4 = nn.ConvTranspose2d(
            features * 24, features * 8, kernel_size=2, stride=2  #Himu features * 16
        )
        self.Att4 = Attention_block(F_g = 8 * features, F_l = 8 * features, F_int = 4 * features)
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.Att3 = Attention_block(F_g = 4 * features, F_l = 4 * features, F_int = 2 * features)
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.Att2 = Attention_block(F_g = 2 * features, F_l = 2 * features, F_int = features)
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.Att1 = Attention_block(F_g = features, F_l = features, F_int = int(features/2))

        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.Sequential(                    
                   nn.Linear(1536, 1536),
                   nn.ReLU(),
                   nn.Linear(1536, 1536))


    def forward(self, x, feats):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        feat_all = feats.unsqueeze(2).unsqueeze(2).repeat(1,1,bottleneck.shape[2],bottleneck.shape[3])
        combine_feat = F.softmax(torch.matmul(bottleneck,feat_all), dim=1) * feat_all 
        combine_feat = combine_feat + bottleneck
        bottleneck = self.mlp(combine_feat.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous() + combine_feat
        #bottleneck = torch.cat((bottleneck, feat_all), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4 = self.Att4(g = dec4, x = enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.Att3(g = dec3, x = enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.Att2(g = dec2, x = enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.Att1(g = dec1, x = enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
