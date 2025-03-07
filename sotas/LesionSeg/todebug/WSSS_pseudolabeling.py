"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import numpy as np

import torch.utils.data

"""
代码还没有debug,
伪标签生成阶段包括
伪标签生成阶段由三个基本部分组成，如图 2 所示
图 2 所示：(1) 从给定的输入图像中合成伪健康图像。
(1) 从给定的输入图像中合成伪健康图像，该图像可能是患病图像，也可能是健康图像。
(2) 利用图像异常特征进行分类，生成 CAM。
(2) 利用图像级类别标签对异常特征进行分类，生成 CAM，以及
(3) 通过迭代改进学习生成最终的伪标签。
"""

class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True
        
    def inference(self, image_tensor):
        with torch.no_grad():
            # self.netg.eval()
            # print(image_tensor.is_cuda)
            self.input.resize_(image_tensor.size()).copy_(image_tensor)
            self.fake, latent_i, latent_o = self.netg(self.input)
            return self.fake.data

            
##
# origin image + 
class Ganomaly(BaseModel):
    def __init__(self, opt, path):
        super(Ganomaly, self).__init__(opt)
        # cpu
        self.device = torch.device(self.opt.device)
        self.netg = NetG(self.opt).to(self.device)
        self.netg.apply(weights_init)
        
        with torch.no_grad():
            self.netg.eval()
            try:
                pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")

        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)

""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
from bdb import set_trace
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))    
        else:
            output = self.main(input)
        # import pdb; pdb.set_trace()

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main
        
    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))    
        else:
            output = self.main(input)
        return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        # print(latent_i[0])
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
from utils.utils import (
    OrgLabels,
    SegmentationModelOutputWrapper,
    post_process_cam,
    get_num_classes,
)
from pytorch_grad_cam import GradCAM
import torch.nn.functional as F

"""
5 labels: image -> 2 / 5 (irf, ez)  -> class_prob > 0.5
[0.1, 0.8, 0.1, 0.8, 1]
cam * 5 -> prob (0-1)
Muli(cam) -> 0
Sum(cam5) -> 1
filter cam5 -> cam2 -> Sum() -> normalize
"""


def refine_input_by_cam(
    device, multi_task_model, input_tensor, mask_tensor, aug_smooth=False
):
    multi_task_model.eval()
    with torch.no_grad():
        cls_outputs = multi_task_model(input_tensor)
    batch_cam_masks = []
    target_layers = multi_task_model.get_cam_target_layers()
    wrap_model = SegmentationModelOutputWrapper(multi_task_model)
    # TODO: replace the range with 'BackGround' label not the last position case
    for cls in range(get_num_classes()):
        targets = [ClassifierOutputTarget(cls)] * len(
            input_tensor
        )  # for all in batch return the current class cam
        with GradCAM(
            model=wrap_model, use_cuda=device, target_layers=target_layers
        ) as cam:
            batch_grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=False,
                aug_smooth=aug_smooth,
            )
        batch_cam_masks.append(batch_grayscale_cam)  # cls, [batch, w, h]
    batch_cam_masks = np.transpose(
        np.stack(batch_cam_masks), (1, 0, 2, 3)
    )  # np: [batch, cls, w, h]

    updated_input_tensor = input_tensor.clone()
    for batch_idx, singel_cam_masks in enumerate(batch_cam_masks):
        # curr_preds = single_batch_pred # (cls) 0/1 values as cls_outputs threshold by 0.5
        curr_preds = cls_outputs[batch_idx]  # classification probability
        norm_cams = torch.from_numpy(post_process_cam(singel_cam_masks)).to(device)
        """CAM heat map * class probability"""
        target_classes_cam = [
            class_cam * curr_preds[cls_i] for cls_i, class_cam in enumerate(norm_cams)
        ]
        # sum the cams for predicted classes
        sum_masks = sum(target_classes_cam)  # (w, h)
        # normalize the above 'attention map' to 0-1
        min, max = sum_masks.min(), sum_masks.max()
        sum_masks.add_(-min).div_(max - min + 1e-5)
        soft_apply = sum_masks.unsqueeze(0).repeat(3, 1, 1)  # (3, w, h)
        """ BackGround CAM * normalized CAM * Original Image. does norm -> multiply order matter? """
        num_channels = 3  # image.shape[1]
        for s in range(0, num_channels, 3):
            inputs_after_soft_addon = (
                soft_apply
                * input_tensor[
                    batch_idx,
                    s : s + 3,
                ]
            )  # [3, w, h]
            # normilize the input image after addon soft map on origin input (both origin & gan)
            soft_min, soft_max = (
                inputs_after_soft_addon.min(),
                inputs_after_soft_addon.max(),
            )
            inputs_after_soft_addon.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
            updated_input_tensor[
                batch_idx,
                s : s + 3,
            ] = inputs_after_soft_addon
        # import torchvision.utils as vutils
        # vutils.save_image(updated_input_tensor.reshape(3,3,512,512), 'test.png', normalize=True, scale_each=True)
    return updated_input_tensor


def get_pseudo_label(params, multi_task_model):
    retinal_mask, input_tensor, cls_labels, args = (
        params["mask"].cpu().numpy(),
        params["input_tensor"],
        params["cls_labels"].cpu().numpy(),
        params["args"],
    )
    pseudo_labels = []
    batch_cam_masks = []
    target_layers = (
        multi_task_model.get_cam_target_layers()
    )  # .module. if use dataparallel
    wrap_model = SegmentationModelOutputWrapper(multi_task_model)

    for cls in range(get_num_classes()):
        targets = [ClassifierOutputTarget(cls)] * len(
            input_tensor
        )  # for all in batch return the current class cam
        with GradCAM(
            model=wrap_model, use_cuda=args.device, target_layers=target_layers
        ) as cam:
            batch_grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=False,
                aug_smooth=args.aug_smooth,
            )
        batch_cam_masks.append(batch_grayscale_cam)  # cls, [batch, w, h]
    batch_cam_masks = np.transpose(
        np.stack(batch_cam_masks), (1, 0, 2, 3)
    )  # np: [batch, cls, w, h]

    for singel_cam_masks, single_retinal_mask, cls_label in zip(
        batch_cam_masks, retinal_mask, cls_labels
    ):
        norm_cams = post_process_cam(
            singel_cam_masks, single_retinal_mask[0]
        )  # [cls, w, h]
        for i in range(get_num_classes()):
            if cls_label[i] == 0:
                norm_cams[i] = 0
        bg_score = [np.ones_like(norm_cams[0]) * args.out_cam_pred_alpha]
        pred_with_bg_score = np.concatenate((bg_score, norm_cams))  # [cls+1, w, h]
        """Generate psuedo label by gt labels"""
        pred_labels = np.argmax(pred_with_bg_score, axis=0)  # [0 - num_class]
        pseudo_labels.append(pred_labels)
    import pdb

    pdb.set_trace()
    return torch.LongTensor(pseudo_labels)

def main(args):
    """
    主函数：实现伪标签生成的完整流程
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化GANomaly模型
    class Options:
        def __init__(self):
            self.device = 'cuda'
            self.batchsize = 1
            self.isize = 256  # 输入图像大小
            self.nc = 3      # 输入通道数
            self.nz = 100    # 潜在向量维度
            self.ngf = 64    # 生成器特征图数量
            self.extralayers = 0
            self.ngpu = 1
            self.manualseed = -1

    opt = Options()
    ganomaly_model = Ganomaly(opt, args.ganomaly_weights_path)
    ganomaly_model.netg.eval()
    
    # 2. 加载分类模型
    multi_task_model = args.model  # 假设模型已经在args中定义
    multi_task_model.to(device)
    multi_task_model.eval()
    
    def process_batch(input_tensor, mask_tensor, cls_labels):
        """处理单个批次的数据"""
        # 生成健康图像
        fake_healthy = ganomaly_model.inference(input_tensor)
        
        # 提取异常特征（原始图像和生成图像的差异）
        anomaly_features = torch.abs(input_tensor - fake_healthy)
        
        # 使用CAM进行特征提取和优化
        refined_features = refine_input_by_cam(
            device, 
            multi_task_model, 
            anomaly_features, 
            mask_tensor, 
            aug_smooth=args.aug_smooth
        )
        
        # 生成伪标签
        params = {
            "mask": mask_tensor,
            "input_tensor": refined_features,
            "cls_labels": cls_labels,
            "args": args
        }
        pseudo_labels = get_pseudo_label(params, multi_task_model)
        
        return fake_healthy, anomaly_features, refined_features, pseudo_labels
    
    # 3. 处理数据
    results = {
        'fake_healthy': [],
        'anomaly_features': [],
        'refined_features': [],
        'pseudo_labels': []
    }
    
    with torch.no_grad():
        for batch_idx, (input_tensor, mask_tensor, cls_labels) in enumerate(args.dataloader):
            input_tensor = input_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            cls_labels = cls_labels.to(device)
            
            fake_healthy, anomaly_features, refined_features, pseudo_labels = process_batch(
                input_tensor, mask_tensor, cls_labels
            )
            
            # 收集结果
            results['fake_healthy'].append(fake_healthy.cpu())
            results['anomaly_features'].append(anomaly_features.cpu())
            results['refined_features'].append(refined_features.cpu())
            results['pseudo_labels'].append(pseudo_labels.cpu())
            
            if batch_idx % args.log_interval == 0:
                print(f'Processed batch: {batch_idx}/{len(args.dataloader)}')
    
    return results


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision.utils as vutils
    import os
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='WSSS Pseudo-labeling')
    parser.add_argument('--ganomaly_weights_path', type=str, default='./weights/ganomaly.pth',
                        help='GANomaly预训练模型路径')
    parser.add_argument('--aug_smooth', action='store_true', help='是否使用CAM增强平滑')
    parser.add_argument('--out_cam_pred_alpha', type=float, default=0.2, 
                        help='背景类别的权重')
    parser.add_argument('--log_interval', type=int, default=1, help='日志打印间隔')
    parser.add_argument('--save_dir', type=str, default='./results', 
                        help='结果保存路径')
    args = parser.parse_args()
    
    # 创建测试数据
    batch_size = 4
    img_size = 256
    num_classes = 3
    
    # 模拟输入数据
    fake_input = torch.randn(batch_size, 3, img_size, img_size)
    fake_mask = torch.zeros(batch_size, 1, img_size, img_size)
    fake_cls_labels = torch.randint(0, 2, (batch_size, num_classes))
    
    # 创建数据加载器
    dataset = TensorDataset(fake_input, fake_mask, fake_cls_labels)
    args.dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 模拟多任务模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, num_classes, 1)
        
        def forward(self, x):
            return self.conv(x)
        
        def get_cam_target_layers(self):
            return [self.conv]
    
    args.model = DummyModel()
    
    try:
        # 运行主函数
        results = main(args)
        
        # 保存一些结果示例
        for key in results:
            if len(results[key]) > 0:
                sample = torch.cat(results[key][:4], dim=0)
                vutils.save_image(
                    sample,
                    os.path.join(args.save_dir, f'{key}.png'),
                    normalize=True,
                    nrow=2
                )
        print("Results saved to:", args.save_dir)
        
    except Exception as e:
        print("Error occurred during execution:")
        print(e)
        import traceback
        traceback.print_exc()