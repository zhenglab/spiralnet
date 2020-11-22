import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
import imp
import numpy as np
from .networks import Dis_Imagine, ImagineNet, Outpad, D, PatchD, DenseD
from .loss import ColorLoss, PerceptualLoss, StyleLoss, AdversarialLoss, VGG19, cal_gradient_penalty, SceneLoss
from ..utils import template_match, Adam16


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        self.device = config.DEVICE
        self.imagine_g_weights_path = os.path.join(config.PATH, 'imagine_g.pth')
        self.imagine_d_weights_path = os.path.join(config.PATH, 'imagine_d.pth')

        self.slice_g_weights_path = os.path.join(config.PATH, 'slice_g.pth')
        self.slice_d_weights_path = os.path.join(config.PATH, 'slice_d.pth')
        self.slice_patch_d_weights_path = os.path.join(config.PATH, 'slice_patchD.pth')

    def load(self):
        if self.name == 'ImagineGAN':
            if os.path.exists(self.imagine_g_weights_path):
                print('Loading %s Model ...' % self.name)

                g_data = torch.load(self.imagine_g_weights_path)
                self.g.load_state_dict(g_data['params'])
                self.iteration = g_data['iteration']

            if os.path.exists(self.imagine_d_weights_path):
                d_data = torch.load(self.imagine_d_weights_path)
                self.d.load_state_dict(d_data['params'])

        if self.name == 'SliceGAN':
            if os.path.exists(self.slice_g_weights_path):
                print('Loading %s Model ...' % self.name)

                g_data = torch.load(self.slice_g_weights_path)
                self.g.load_state_dict(g_data['params'])
                self.iteration = g_data['iteration']

            if os.path.exists(self.slice_d_weights_path):
                d_data = torch.load(self.slice_d_weights_path)
                self.d.load_state_dict(d_data['params'])

            if os.path.exists(self.slice_patch_d_weights_path):
                patchd_data = torch.load(self.slice_patch_d_weights_path)
                self.patch_d.load_state_dict(patchd_data['params'])
        
    
    def save(self):
        print('\nSaving %s...\n' % self.name)
        if self.name == 'ImagineGAN':
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path)
            torch.save({'params': self.d.state_dict()}, self.imagine_d_weights_path)

        if self.name == 'SliceGAN':
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.slice_g_weights_path)
            torch.save({'params': self.d.state_dict()}, self.slice_d_weights_path)
            torch.save({'params': self.patch_d.state_dict()}, self.slice_patch_d_weights_path)


class ImagineModel(BaseModel):
    def __init__(self, config):
        super(ImagineModel, self).__init__('ImagineGAN', config)        
        self.catmask = config.CATMASK
        if self.catmask:
            g = ImagineNet(in_channels=7, out_channels=3)
            d = Dis_Imagine(in_channels=6)        
        else :
            g = ImagineNet(in_channels=3, out_channels=3, expand=True)
            d = Dis_Imagine(in_channels=6)    

        color_loss = ColorLoss()
        adversarial_loss = AdversarialLoss()
        l1_loss = nn.L1Loss()

        vgg19 = VGG19(1)
        content_loss = PerceptualLoss(vgg19, weights=[1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.add_module('g', g)
        self.add_module('d', d)

        self.add_module('content_loss', content_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('color_loss', color_loss)
        self.add_module('l1_loss', l1_loss)

        
        self.g_optimizer = Adam16(params=g.parameters(), lr=float(config.G_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0,
                           eps=1e-8)
        self.d_optimizer = Adam16(params=d.parameters(), lr=float(config.D_LR), betas=(config.BETA1, config.BETA2), weight_decay=0.0,
                           eps=1e-8)
        
    def process(self, data, pdata, pos, half_fmask, temp_mask, z):
        self.iteration += 1
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        if self.catmask:
            o = self.g(torch.cat((pdata, half_fmask, z), dim=1))
        else:
            o = self.g(pdata)

        g_loss = 0
        d_loss = 0
        d_real = data
        d_fake = o.detach()
        if self.catmask:
            d_real_, d_real_feat = self.d(torch.cat((d_real, pdata), dim=1))
            d_fake_, d_fake_feat = self.d(torch.cat((d_fake, pdata), dim=1))
            d_real_l = self.adversarial_loss(d_real_, True, True)
            d_fake_l = self.adversarial_loss(d_fake_, False, True)
            d_loss += (d_real_l + d_fake_l) / 2
            g_fake, _ = self.d(torch.cat((o, pdata), dim=1))

        else:
            d_real_, d_real_feat = self.d(torch.cat((d_real, pdata), dim=1))
            d_fake_, d_fake_feat = self.d(torch.cat((d_fake, pdata), dim=1))
            d_real_l = self.adversarial_loss(d_real_, True, True)
            d_fake_l = self.adversarial_loss(d_fake_, False, True)
            d_loss += (d_real_l + d_fake_l) / 2        
            g_fake, _ = self.d(torch.cat((o, pdata), dim=1))        

        g_adv = self.adversarial_loss(g_fake, True, False) * self.config.G1_ADV_LOSS_WEIGHT
        g_loss += g_adv
        g_content_loss, g_class_loss = self.content_loss(o, data)
        g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
        g_loss += g_content_loss

        g_color_loss = self.color_loss(o, data)
        g_color_loss = g_color_loss * self.config.G1_COLOR_LOSS_WEIGHT
        g_loss += g_color_loss
        
        logs = [
            ("l_d", d_loss.item()),           
            ("l_g_adv", g_adv.item()),
            ("l_g_con", g_content_loss.item()),
            ("l_color", g_color_loss.item())
        ]
        return o, d_loss, g_loss, logs
    
    def forward(self, pdata, half_fmask=None, z=None):
        if self.catmask:
            o = self.g(torch.cat((pdata, half_fmask, z), dim=1))
        else:
            o = self.g(pdata)
        return o        

    def backward(self, d_loss, g_loss):
        d_loss.backward()
        self.d_optimizer.step()
        g_loss.backward()
        self.g_optimizer.step()


class SliceModel(BaseModel):
    def __init__(self, config):
        super(SliceModel, self).__init__('SliceGAN', config)

        if self.config.DATATYPE == 1:
            self.out_size = (256,256)
        else:
            self.out_size = (512,256)
        self.each = config.SLICE
        g = Outpad(out_size=self.out_size, device=config.DEVICE, each=self.each)
        d = DenseD()
        patch_d = PatchD(in_channels=3)
        vgg19 = VGG19(2)

        color_loss = nn.CosineSimilarity(dim=3)
        adversarial_loss = AdversarialLoss()
        l1_loss = nn.L1Loss()
        style_loss = StyleLoss(vgg19)

        self.add_module('g', g)
        self.add_module('d', d)
        self.add_module('patch_d', patch_d)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('color_loss', color_loss)
        self.add_module('style_loss', style_loss)

        self.g_optimizer = optim.Adam(params=g.parameters(), lr=float(config.G_LR), betas=(config.BETA1, config.BETA2))
        self.d_optimizer = optim.Adam(params=d.parameters(), lr=float(config.D_LR), betas=(config.BETA1, config.BETA2))
        self.patch_p_optimizer = optim.Adam(params=patch_d.parameters(), lr=float(config.D_LR),
                                            betas=(config.BETA1, config.BETA2))

    def process(self, data, pdata, fm, position, stage, mask):
        self.iteration += 1
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.patch_p_optimizer.zero_grad()

        o, coord_all = self.g(pdata, data, fm, position, stage, post_train=True)

        g_loss = 0
        d_loss = 0
        patch_d_loss = 0
        coord_loss = 0
        coord_loss_w = 0.05
        coord_loss_i_sum = 0
        count_true = 0

        d_real = data
        d_fake = o.detach()
        d_real_ = self.d(d_real)
        d_fake_global = self.d(o.detach())

        d_real_l = self.adversarial_loss(d_real_, True, True)
        d_fake_g = self.adversarial_loss(d_fake_global, False, True)
        d_loss += (d_real_l + d_fake_g) / 2
        g_fake_global = self.d(o)
        
        g_g_loss = self.adversarial_loss(g_fake_global, True, False) * self.config.G2_ADV_LOSS_WEIGHT
        g_loss += g_g_loss
        
        g_l1_loss = self.l1_loss(o, data) * self.config.G2_L1_LOSS_WEIGHT
        g_loss += g_l1_loss
        
        g_color_loss = 1 - torch.mean(self.color_loss(o.permute(0, 2, 3, 1), data.permute(0, 2, 3, 1)))
        g_color_loss = g_color_loss * self.config.G1_COLOR_LOSS_WEIGHT
        g_loss += g_color_loss

        coord_num = len(coord_all)

        for i in range(coord_num):
            if coord_all[i] == None:
                continue

            c_x1, c_y1, c_x2, c_y2 = coord_all[i][0], coord_all[i][1], coord_all[i][2], coord_all[i][3]
            if i % 4 == 0:
                slice_real = d_real[:, :, c_y1:c_y2, c_x1:c_x1 + 64]
                slice_fake = o[:, :, c_y1:c_y2, c_x1:c_x1 + 64]
            if i % 4 == 1:
                slice_real = d_real[:, :, c_y1:c_y1 + 64, c_x1:c_x2]
                slice_fake = o[:, :, c_y1:c_y1 + 64, c_x1:c_x2]
            if i % 4 == 2:
                slice_real = d_real[:, :, c_y1:c_y2, c_x2 - 64:c_x2]
                slice_fake = o[:, :, c_y1:c_y2, c_x2 - 64:c_x2]
            if i % 4 == 3:
                slice_real = d_real[:, :, c_y2 - 64:c_y2, c_x1:c_x2]
                slice_fake = o[:, :, c_y2 - 64:c_y2, c_x1:c_x2]

            d_reals, _ = self.patch_d(slice_real)
            d_fakes, _ = self.patch_d(slice_fake)

            d_real_slice = self.adversarial_loss(d_reals, True, True)
            d_fake_slice = self.adversarial_loss(d_fakes, False, True)
            g_fake_slice = self.adversarial_loss(d_fakes, True, False)
            patch_d_loss += self.config.G2_ADV_LOSS_WEIGHT * (d_real_slice + d_fake_slice) / 2

            coord_loss_i_sum += self.config.G2_ADV_LOSS_WEIGHT * g_fake_slice

        g_loss += coord_loss_i_sum

        g_fake_local = self.d(o * mask + data * (1 - mask))
        g_l_loss = self.adversarial_loss(g_fake_local, True, False) * self.config.G2_ADV_LOSS_WEIGHT
        g_adv_loss = g_l_loss
        g_loss += g_adv_loss

        g_style_loss, g_mrf_loss = self.style_loss(o, data)
        g_loss += g_mrf_loss * self.config.G2_MRF_LOSS_WEIGHT

        logs = [
            ("l_d", d_loss.item()),
            ("l_pd_slice", patch_d_loss.item()),
            ("l_pd_coord", coord_loss_i_sum.item()),
            ("l_gg_adv", g_g_loss.item()),
            ("l_gl_adv", g_l_loss.item()),
            ("l_g_adv", g_adv_loss.item()),
            ("l_g_mrf", g_mrf_loss.item()),
            ("l_g_l1", g_l1_loss.item()),
            ("l_g_col", g_color_loss.item())
        ]

        return o, d_loss, g_loss, patch_d_loss, logs

    def forward(self, x, data, fm, position, stage, post_train=True):
        o, coord = self.g(x, data, fm, position, stage, post_train=True)
        return o, coord

    def backward(self, d_loss, g_loss, patch_d_loss):

        patch_d_loss.backward(retain_graph=True)
        self.patch_p_optimizer.step()

        d_loss.backward()
        self.d_optimizer.step()

        g_loss.backward()
        self.g_optimizer.step()