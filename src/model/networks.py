import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from collections import OrderedDict
from .loss import VGG19


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, use_dropout=False):
        super(ResnetBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
         ]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
    
class ImagineNet(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True, in_channels=3, out_channels=3, expand=True):
        super(ImagineNet, self).__init__()
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)
        
        self.outer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )
        self.tanh = nn.Tanh()
            
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.outer(x)
        x = (self.tanh(x) + 1) / 2
        return x


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.batch_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.conv = spectral_norm(nn.Conv2d(norm_nc, norm_nc, kernel_size=3, padding=1))

        self.shared = nn.Sequential(nn.Conv2d(label_nc, 128, kernel_size=3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def norm(self, x, parsing):
        bn = self.batch_norm(x)

        parsing = F.interpolate(parsing, size=x.size()[2:], mode='nearest')
        tmp_feature = self.shared(parsing)
        gamma = self.gamma(tmp_feature)
        beta = self.beta(tmp_feature)

        (b_g, c_g, h_g, w_g) = gamma.shape
        (b_b, c_b, h_b, w_b) = beta.shape
        gamma = torch.mean(gamma.view(b_g, c_g, -1), dim=2).view(b_g, c_g, 1, 1)
        beta = torch.mean(beta.view(b_b, c_b, -1), dim=2).view(b_b, c_b, 1, 1)
        out = bn * (1 + gamma) + beta

        return out

    def forward(self, x, segmap):
        x_tmp = x

        # step 1
        norm = self.norm(x, segmap)
        # step 2
        act = self.actvn(norm)
        dx = self.conv(act)
        out = x_tmp + dx
        return act


class Gen(nn.Module):
    def __init__(self, inc, outc, mid, n_block=6):
        super(Gen, self).__init__()
        inchannel = inc + 8

        vgg19 = VGG19(2)
        self.add_module('vgg_layer', vgg19)
        self.inmodel_c1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(inchannel, mid, kernel_size=7, padding=0),
            nn.InstanceNorm2d(mid),
            nn.ReLU(True),
        )

        self.inmodel = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(inc, mid, kernel_size=7, padding=0),
            nn.InstanceNorm2d(mid),
            nn.ReLU(True),
        )

        self.downblock_c1 = nn.Sequential(
            nn.Conv2d(mid, mid * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(mid * 2),
            nn.ReLU(True),

            nn.Conv2d(mid * 2, mid * 2 * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(mid * 2 * 2),
            nn.ReLU(True),
        )

        self.inmodel_c2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(inc, mid, kernel_size=7, padding=0),
            nn.InstanceNorm2d(mid),
            nn.ReLU(True),
        )

        self.downblock_c2 = nn.Sequential(
            nn.Conv2d(mid, mid * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(mid * 2),
            nn.ReLU(True),

            nn.Conv2d(mid * 2, mid * 2 * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(mid * 2 * 2),
            nn.ReLU(True),

            nn.Conv2d(mid * 2 * 2, mid * 2 * 2 * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(mid * 2 * 2 * 2),
            nn.ReLU(True),
        )

        blocks = []
        for i in range(n_block):
            resblocks = ResnetBlock(mid * 2 * 2, 2)
            blocks.append(resblocks)
        self.resblock = nn.Sequential(*blocks)

        self.upblock1 = nn.Sequential(
            nn.ConvTranspose2d(mid * 2 * 2, mid * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(mid * 2),
            nn.ReLU(True)
        )

        self.upblock2 = nn.Sequential(
            nn.ConvTranspose2d(mid * 2, mid, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(mid),
            nn.ReLU(True),
        )

        self.outmodel = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(mid, outc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

        self.up_1 = SPADE(4 * 64, 3)
        self.up_2 = SPADE(2 * 64, 3)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def AdaIN(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def add_pad(self, x, H, W):
        if H:
            add_padding = nn.ReflectionPad2d((0, 0, H, 0))
            x = add_padding(x)
        if W:
            add_padding = nn.ReflectionPad2d((W, 0, 0, 0))
            x = add_padding(x)
        return x

    def forward(self, c1, pres, s, coord):
        H, W = c1.shape[2], c1.shape[3]

        c1 = self.inmodel(c1)
        c1_in = self.downblock_c1(c1)

        s_vgg = self.vgg_layer(s)
        s_in = s_vgg['relu3_4']

        x = self.AdaIN(c1_in, s_in)
        x = self.resblock(x)

        x1 = self.up_1(x, pres)
        x1 = self.upblock1(x1)
        x2 = self.up_2(x1, pres)
        x2 = self.upblock2(x2)
        x = F.interpolate(x2, (H, W), mode='bilinear',align_corners=False)

        x = self.outmodel(x)
        return x


class SpiralNet(BaseNetwork):
    def __init__(self, out_size, device, ks=3, each=16):
        super(SpiralNet, self).__init__()

        self.ks, self.out_size, self.device, self.each = ks, out_size, device, each

        self.G = Gen(inc=3, outc=3, mid=64)

        self.init_weights()
        self.strid = self.each  # must divide by 4

        self.wmax, self.hmax = self.out_size[0], self.out_size[1]

    def cat_sf(self, x, sfdata, p, direct):
        h, w = x.shape[2], x.shape[3]

        if direct == 0:
            crop_sf = sfdata[:, :, p[1]:p[1] + h, p[0]:p[0] + w]
            slice_in = torch.cat((crop_sf, x), dim=3)
            return crop_sf
        elif direct == 1:
            crop_sf = sfdata[:, :, p[1]:p[1] + h, p[0]:p[0] + w]
            slice_in = torch.cat((crop_sf, x), dim=2)
            return crop_sf
        elif direct == 2:
            crop_sf = sfdata[:, :, p[1] - h:p[1], p[0] - w:p[0]]
            slice_in = torch.cat((x, crop_sf), dim=3)
            return crop_sf
        elif direct == 3:
            crop_sf = sfdata[:, :, p[1] - h:p[1], p[0] - w:p[0]]
            slice_in = torch.cat((x, crop_sf), dim=2)
            return crop_sf

    def sliceOperator_1(self, x, direct):
        # direct => l/u/r/d: 0,1,2,3
        h, w = x.shape[2], x.shape[3]
        if direct == 0:
            return x[:, :, :, :self.each]
        if direct == 1:
            return x[:, :, :self.each, :]
        if direct == 2:
            return x[:, :, :, -self.each:]
        if direct == 3:
            return x[:, :, -self.each:, :]

    def extrapolateOperator(self, x, direct, conv_data, loc):
        cat_edge = self.each
        if direct == 0:
            if loc < self.each:
                cat_edge = loc - 0
            x = torch.cat((conv_data[:, :, :, :cat_edge], x), dim=3)
        if direct == 1:
            if loc < self.each:
                cat_edge = loc - 0
            x = torch.cat((conv_data[:, :, :cat_edge, :], x), dim=2)
        if direct == 2:
            if loc > (self.wmax - self.each):
                cat_edge = self.wmax - loc
            x = torch.cat((x, conv_data[:, :, :, -cat_edge:]), dim=3)
        if direct == 3:
            if loc > (self.hmax - self.each):
                cat_edge = self.hmax - loc
            x = torch.cat((x, conv_data[:, :, -cat_edge:, :]), dim=2)

        return x

    def sliceOperator_from(self, x, direct, gt, coord):
        h, w = x.shape[2], x.shape[3]
        x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
        if direct == 0:
            return gt[:, :, y1:y1 + h, x1:x1 + self.each]
        if direct == 1:
            return gt[:, :, y1:y1 + self.each, x1:x1 + w]
        if direct == 2:
            return gt[:, :, y2 - h:y2, x2 - self.each:x2]
        if direct == 3:
            return gt[:, :, y2 - self.each:y2, x1:x1 + w]

    def sliceGenerator(self, stg, coord, inits, subimage, fm, each, direct=0, post_train=False):
        iner = subimage
        if direct == 0:
            if coord[0] > 0:
                loc_x1 = coord[0]
                coord[0] -= each
                coord[0] = coord[0] if coord[0] > 0 else 0
                dir = [1, 0, 0, 0]
                stg = stg + dir
                c_left = self.sliceOperator_1(iner, direct)
                if not post_train:
                    styslice = self.sliceOperator_from(c_left, direct, gt, coord)
                else:
                    styslice = self.sliceOperator_1(c_left, direct)
                slice_in = self.cat_sf(c_left, fm, (coord[0], coord[1]), direct)
                l = self.G(slice_in, styslice, inits, stg)
                iner = self.extrapolateOperator(iner, direct, l, loc_x1)
                return iner, coord[0], coord[1]

            else:
                return iner, coord[0], coord[1]

        elif direct == 1:
            if coord[1] > 0:
                loc_y1 = coord[1]
                coord[1] -= each
                coord[1] = coord[1] if coord[1] > 0 else 0
                dir = [0, 1, 0, 0]
                stg = stg + dir
                c_up = self.sliceOperator_1(iner, direct)
                if not post_train:
                    styslice = self.sliceOperator_from(c_up, direct, gt, coord)
                else:
                    styslice = self.sliceOperator_1(c_up, direct)
                slice_in = self.cat_sf(c_up, fm, (coord[0], coord[1]), direct)
                u = self.G(slice_in, styslice, inits, stg)
                iner = self.extrapolateOperator(iner, direct, u, loc_y1)
                return iner, coord[0], coord[1]
            else:
                return iner, coord[0], coord[1]

        elif direct == 2:
            if coord[2] < self.wmax:
                loc_x2 = coord[2]
                coord[2] += each
                coord[2] = coord[2] if coord[2] < self.wmax else self.wmax
                dir = [0, 0, 1, 0]
                stg = stg + dir
                c_right = self.sliceOperator_1(iner, direct)
                if not post_train:
                    styslice = self.sliceOperator_from(c_right, direct, gt, coord)
                else:
                    styslice = self.sliceOperator_1(c_right, direct)
                slice_in = self.cat_sf(c_right, fm, (coord[2], coord[3]), direct)
                r = self.G(slice_in, styslice, inits, stg)
                iner = self.extrapolateOperator(iner, direct, r, loc_x2)
                return iner, coord[2], coord[3]
            else:
                return iner, coord[2], coord[3]

        elif direct == 3:
            if coord[3] < self.hmax:
                loc_y2 = coord[3]
                coord[3] += each
                coord[3] = coord[3] if coord[3] < self.hmax else self.hmax
                dir = [0, 0, 0, 1]
                stg = stg + dir
                c_down = self.sliceOperator_1(iner, direct)
                if not post_train:
                    styslice = self.sliceOperator_from(c_down, direct, gt, coord)
                else:
                    styslice = self.sliceOperator_1(c_down, direct)
                slice_in = self.cat_sf(c_down, fm, (coord[2], coord[3]), direct)
                d = self.G(slice_in, styslice, inits, stg)
                iner = self.extrapolateOperator(iner, direct, d, loc_y2)
                return iner, coord[2], coord[3]
            else:
                return iner, coord[2], coord[3]

    def forward(self, x, gt, fm, position, stage, post_train=False):
        x1, y1, x2, y2 = position[0][0].item(), position[0][1].item(), position[1][0].item(), position[1][1].item()
        inits = x
        coord_all = []
        post_train = post_train

        for st in range(int(stage)):
            gen = x
            stg = [0, 0, 0, 0]
            gen, x1, y1 = self.sliceGenerator(stg, [x1, y1, x2, y2], inits, gen, fm, self.each, direct=0,
                                        post_train=post_train)
            coord_all.append([x1, y1, x2, y2])
            gen, x1, y1 = self.sliceGenerator(stg, [x1, y1, x2, y2], inits, gen, fm, self.each, direct=1,
                                        post_train=post_train)
            coord_all.append([x1, y1, x2, y2])
            gen, x2, y2 = self.sliceGenerator(stg, [x1, y1, x2, y2], inits, gen, fm, self.each, direct=2,
                                        post_train=post_train)
            coord_all.append([x1, y1, x2, y2])
            gen, x2, y2 = self.sliceGenerator(stg, [x1, y1, x2, y2], inits, gen, fm, self.each, direct=3,
                                        post_train=post_train)
            coord_all.append([x1, y1, x2, y2])

            x = gen

        return x, coord_all


class D(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(D, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]


class PatchD(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(PatchD, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.stage, self.direction = 4, 4

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def aux_module(self, stage, direction, inchan, aux_in):
        aux_layer_1 = nn.Linear(inchan, stage * direction).cuda()
        aux_soft = nn.Softmax(dim=1).cuda()
        aux = aux_layer_1(aux_in)
        aux = aux_soft(aux)
        return aux

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        aux_layer = conv3
        aux_in = aux_layer.view(conv3.shape[0], -1)
        inp = aux_in.shape[1]

        outputs = conv4
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv4)

        aux = self.aux_module(self.stage, self.direction, inp, aux_in)

        return outputs, aux


class Dis_Imagine(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Dis_Imagine, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()
     
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseD(nn.Module):
    def __init__(self, growth_rate=32, block_config=(3, 3, 3),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseD, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=False), True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = torch.sigmoid(out)
        return out