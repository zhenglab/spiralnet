import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import Dataset
from .model.model import SliceModel
from .utils import Progbar, create_dir, stitch_images, imsave, template_match, imcopy
from PIL import Image
from tensorboardX import SummaryWriter
from .model.networks import ImagineNet
from .metrics import PSNR
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import cv2
from glob import glob
from ntpath import basename
from imageio import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray


class Slice():
    def __init__(self, config):
        self.config = config

        self.model_name = 'SliceModel'
        self.Model = SliceModel(config).to(config.DEVICE)
        
        self.psnr = PSNR(255.0).to(config.DEVICE)

        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.MASK_FLIST, augment=False, training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.MASK_FLIST, augment=False, training=True)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        
        self.samples_path = os.path.join(config.PATH, 'samples_slice')

        if config.CENTER == 1:
            self.results_path = os.path.join(config.PATH, 'result_slice_c')
        else:
            self.results_path = os.path.join(config.PATH, 'result_slice')
        
        self.log_file = os.path.join(config.PATH, 'log-' + self.model_name + '.txt')
        
        self.writer = SummaryWriter(os.path.join(config.PATH, 'runs'))
        imagine_g_weights_path = os.path.join(config.PATH, 'imagine_g.pth')
        g_data = torch.load(imagine_g_weights_path)

        if self.config.CATMASK:
            self.imagine_g = ImagineNet(in_channels=7).to(config.DEVICE)
        else:
            self.imagine_g = ImagineNet(in_channels=3).to(config.DEVICE)

        self.imagine_g.load_state_dict(g_data['params'])
        print(self.imagine_g.load_state_dict(g_data['params']))
        imgsize = 256 // 2
        if self.config.DATATYPE != 1:
            imgsize = 512 // 2
        
        self.slice = self.config.SLICE
        self.stage = imgsize // self.slice
        

    def load(self):
        self.Model.load()

    def save(self):
        self.Model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=False,
            shuffle=True,
            pin_memory=True
        )
        epoch = 0
        keep_training = True

        max_iter = int(self.config.MAX_ITERS)
        total = len(self.train_dataset)

        while (keep_training):
            epoch += 1
            probar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            ite = self.Model.iteration
            for it in train_loader:
                self.Model.train()
                data, pdata, fullpdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*it)

                if self.config.CATMASK:
                    sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
                else:
                    sf_data = self.imagine_g(pdata)

                if self.config.DATATYPE == 1:
                    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    pos = pos[0]
                    position = (pos, (pos[0] + 128, pos[1] + 128))
                if self.config.DATATYPE == 2:
                    up = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=False)
                    pos = pos[0]
                    position = (pos, (pos[0] + 256, pos[1] + 256))

                sf_data = up(sf_data)

                outputs, d_loss, g_loss, patch_d_loss, logs = self.Model.process(data, fullpdata, sf_data,
                                                                                               position, self.stage,
                                                                                               fmask_data)
                self.Model.backward(d_loss, g_loss, patch_d_loss)
                psnr = self.psnr(self.postprocess(data), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(data - (outputs))) / torch.sum(data)).float()
                ite = self.Model.iteration

                # ------------------------------------------------------------------------------------
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('g_loss', g_loss.item()))

                logs = [("epoch", epoch), ("iter", ite), ("slice", self.slice)] + logs
                self.writer.add_scalars('Discriminator', {'domaink': d_loss}, epoch)
                self.writer.add_scalars('Generator', {'domaink': g_loss}, epoch)
                self.writer.add_scalars('Detail', self.log2dict(logs), epoch)
                probar.add(len(data), values=[x for x in logs])
                if ite <= max_iter and ite % self.config.INTERVAL == 0:
                    self.log(logs)
                    self.sample()
                    self.save()
        print('\nEnd trainging...')
        self.writer.close()

    def log2dict(self, logs):
        dict = {}
        for i in range(2, len(logs)):
            dict[logs[i][0]] = logs[i][1]
        return dict

    def compare_mae(self, img_true, img_test):
        img_true = img_true.astype(np.float32)
        img_test = img_test.astype(np.float32)
        if np.sum(img_true + img_test) == 0:
            return 1
        return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

    def compute_metric(self, gt_path, output):
        psnr = []
        ssim = []
        mae = []
        names = []
        index = 1

        files = list(glob(gt_path + '/*.jpg')) + list(glob(gt_path + '/*.png'))
        for fn in sorted(files):
            name = basename(str(fn))
            names.append(name)
            pred_name = str(fn)
            img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
            img_pred = (imread(output + '/' + basename(pred_name)) / 255.0).astype(np.float32)

            imgh, imgw = img_gt.shape[0:2]
            img_pred = cv2.resize(img_pred, dsize=(imgw, imgh))
            img_gt = rgb2gray(img_gt)
            img_pred = rgb2gray(img_pred)

            psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
            ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51,multichannel=True))
            mae.append(self.compare_mae(img_gt, img_pred))
            if np.mod(index, 100) == 0:
                print(
                    str(index) + ' images processed',
                    "PSNR: %.4f" % round(np.mean(psnr), 4),
                    "SSIM: %.4f" % round(np.mean(ssim), 4),
                    "MAE: %.4f" % round(np.mean(mae), 4),
                )
            index += 1

        np.savez(self.results_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
        print(
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "SSIM Variance: %.4f" % round(np.var(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE Variance: %.4f" % round(np.var(mae), 4)
        )

    
    def test(self):
        test_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1
        )

        create_dir(self.results_path)
        final_results = os.path.join(self.results_path, 'final_results')
        masks = os.path.join(self.results_path, 'masks')
        state_1_results = os.path.join(self.results_path, 'state_1_results')
        state_rec_results = os.path.join(self.results_path, 'state_rec_results')

        gt_img_path = os.path.join(self.results_path, 'gt_image')
        inner_masks_path = os.path.join(self.results_path, 'inner_masks')

        unknown_final_results = os.path.join(self.results_path, 'unknown_final_results')
        
        create_dir(final_results)
        create_dir(masks)
        create_dir(state_1_results)
        create_dir(state_rec_results)

        create_dir(gt_img_path)
        create_dir(inner_masks_path)
        create_dir(unknown_final_results)
        total = len(self.val_dataset)

        index = 0
        for it in test_loader:
            name, init_name = self.val_dataset.load_name(index)
            index += 1
            data, pdata, fullpdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*it)
            if self.config.CATMASK:
                sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
            else:
                sf_data = self.imagine_g(pdata)

            if self.config.DATATYPE == 1:
                up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                pos = pos[0]
                print("pos: ", pos)
                position = (pos, (pos[0] + 128, pos[1] + 128))
            if self.config.DATATYPE == 2:
                up = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=False)
                pos = pos[0]
                print("pos: ", pos)
                position = (pos, (pos[0] + 256, pos[1] + 256))

            sf_data = up(sf_data)

            o, _ = self.Model(fullpdata, data, sf_data, position, self.stage, post_train=True)
            final = o
            _o_1s, _pos = template_match(fullpdata, sf_data)

            _pos = _pos[0]
            _pos = torch.IntTensor(_pos).cuda()
            print("_pos: ",_pos)
            _position = (_pos, (_pos[0]+128, _pos[1]+128))
     
            o_2, _ = self.Model(fullpdata, data, sf_data, _position, self.stage, post_train=True)
            final_2 = o_2

            pdata = self.postprocess(pdata)[0]
            mask = self.postprocess(mask)[0]
            o_1 = self.postprocess(sf_data)[0]
            final = self.postprocess(final)[0]
            final_2 = self.postprocess(final_2)[0]
            
            imcopy(init_name, gt_img_path)
            imsave(mask, os.path.join(masks, name))
            imsave(pdata, os.path.join(inner_masks_path, name))
            imsave(o_1, os.path.join(state_1_results, name))
            imsave(_o_1s.int(), os.path.join(state_rec_results, name))
            imsave(final, os.path.join(final_results, name))
            imsave(final_2, os.path.join(unknown_final_results, name))

            print(index, name)

        print("start compute metric")

        self.compute_metric(gt_img_path, final_results)

        print('\nEnd test....')
    
    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\r\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def sample(self):

        ite = self.Model.iteration

        its = next(self.sample_iterator)
        data, pdata, fullpdata, pos, fmask_data, half_fmask, mask, z = self.cuda(*its)

        if self.config.CATMASK:
            sf_data = self.imagine_g(torch.cat((pdata, half_fmask, z), dim=1))
        else:
            sf_data = self.imagine_g(pdata)

        if self.config.DATATYPE == 1:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            pos = pos[0]
            position = (pos, (pos[0] + 128, pos[1] + 128))
        if self.config.DATATYPE == 2:
            up = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=False)
            pos = pos[0]
            position = (pos, (pos[0] + 256, pos[1] + 256))
        sf_data = up(sf_data)

        o, _ = self.Model(fullpdata, data, sf_data, position, self.stage)
        _o_1s, _pos = template_match(fullpdata, sf_data)

        image_per_row = 1
        images = stitch_images(
            self.postprocess(mask),
            self.postprocess(sf_data),
            _o_1s.int(),
            self.postprocess(o),
            self.postprocess(data),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path)
        name = os.path.join(path, str(ite).zfill(5) + '.png')
        create_dir(path)
        print('\nSaving training sample images...' + name)
        images.save(name)
