import torch
import torch.nn as nn
import torch.optim as optim
from configuration.config import cfg

class ConvNormAct(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, act_type='relu'):

        super(ConvNormAct, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)]

        layers += [nn.InstanceNorm2d(out_ch)]

        if act_type == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif act_type == 'lrelu':
            layers += [nn.LeakyReLU(0.2)]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):

    def __init__(self, in_ch, device, seq=False, is_extrap=True):

        super(Discriminator, self).__init__()
        self.device = device
        self.seq = seq
        self.is_extrap = is_extrap

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.layer_2 = ConvNormAct(64, 128, kernel_size=4, stride=2, padding=1, act_type='lrelu')
        self.layer_3 = ConvNormAct(128, 256, kernel_size=4, stride=2, padding=1, act_type='lrelu')
        self.layer_4 = ConvNormAct(256, 512, kernel_size=4, stride=1, padding=2, act_type='lrelu')
        self.last_conv = nn.Conv2d(512, 64, kernel_size=4, stride=1, padding=2, bias=False)

    def forward(self, x):
        h = self.layer_1(x)
        h = self.layer_2(h)
        h = self.layer_3(h)
        h = self.layer_4(h)
        return self.last_conv(h)

    def netD_adv_loss(self, real, fake, input_real):
        '''
        :param real: output sequence - ground truth: S_o x B x C x H x W
        :param fake: predictions: S_o x B x C x H x W
        :param input_real: input sequence: S_i x B x C x H x W
        :return:
        '''
        if self.seq: #### for video sequence level
            if self.is_extrap:
                real, fake = self.rearrange_seq(real, fake, input_real, only_fake=False)
            else:
                real, fake = self.rearrange_seq_interp(real, fake, input_real, only_fake=False)
        elif not self.seq: ### do not use input_real
            S, B, C, H, W = fake.size()
            # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
            real = real.contiguous().view(-1, C, H, W)
            fake = fake.contiguous().view(-1, C, H, W)
        pred_fake = self.forward(fake.detach()) # D(G(z))
        pred_real = self.forward(real) # D(x)
        # GAN loss type
        real_label = torch.ones_like(pred_real).to(self.device) # b = 1 (always)
        # E[(D(G(z)) - a)^2] with a = 0 or 1
        loss_fake = torch.mean((pred_fake) ** 2)
        #  E[(D(x) - b)^2]
        loss_real = torch.mean((pred_real - real_label) ** 2)
        # min_D V_LSGAN = 0.5 * (E[(D(x) - b)^2] + E[(D(G(z)) - a)^2])
        loss_D = (loss_real + loss_fake) * 0.5
        return loss_D

    def netG_adv_loss(self, fake, input_real):
        '''
        :param fake: predictions: S_o x B x C x H x W
        :param input_real: input sequence: S_i x B x C x H x W
        :return:
        '''
        S, B, C, H, W = fake.size()
        if self.seq:
            if self.is_extrap:
                fake = self.rearrange_seq(None, fake, input_real, only_fake=True)
            else:
                fake = self.rearrange_seq_interp(None, fake, input_real, only_fake=True)
        elif not self.seq:
            fake = fake.contiguous().view(-1, C, H, W)
        pred_fake = self.forward(fake) # D(G(z))
        real_label = torch.ones_like(pred_fake).to(self.device) # c = 1 or 0
        # min_G V_LSGAN = 0.5 * E[(D(G(z)) - c)^2]
        loss_real = 0.5 * torch.mean((pred_fake - real_label) ** 2)
        return loss_real

    def rearrange_seq(self, real, fake, input_real, only_fake=True):

        t, b, c, h, w = fake.size()
        fake_seqs = []
        for i in range(t):
            fake_seq = torch.cat([input_real[i:, ...], fake[:i + 1, ...]], dim=0)
            fake_seqs += [fake_seq]
        fake_seqs = torch.cat(fake_seqs, dim=0).view(b * t, -1, h, w)
        if only_fake:
            return fake_seqs

        real_seqs = []
        for i in range(t):
            real_seq = torch.cat([input_real[i:, ...], real[:i + 1, ...]], dim=0)
            real_seqs += [real_seq]
        real_seqs = torch.cat(real_seqs, dim=0).view(b * t, -1, h, w)

        return real_seqs, fake_seqs

    def rearrange_seq_interp(self, real, fake, input_real, only_fake=True):

        t, b, c, h, w = fake.size()
        mask = torch.eye(t).float().cuda()
        fake_seqs = []
        for i in range(t):
            reshaped_mask = mask[i].view(1, -1, 1, 1, 1)
            fake_seq = (1 - reshaped_mask) * input_real + reshaped_mask * fake
            fake_seqs += [fake_seq]
        fake_seqs = torch.cat(fake_seqs, dim=0).view(b * t, -1, h, w)

        if only_fake:
            return fake_seqs

        real_seqs = []
        for i in range(t):
            reshaped_mask = mask[i].view(1, -1, 1, 1, 1)
            real_seq = (1 - reshaped_mask) * input_real + reshaped_mask * real
            real_seqs += [real_seq]
        real_seqs = torch.cat(real_seqs, dim=0).view(b * t, -1, h, w)

        return real_seqs, fake_seqs

def create_netD(irregular=cfg.MODEL.IRREGULAR, extrap=cfg.MODEL.EXTRAPOLATION_MODE,
                sample_size=cfg.MODEL.SAMPLE_SIZE, lr=cfg.MODEL.LR):
    # Model
    device = cfg.GLOBAL.DEVICE
    seq_len = sample_size // 2
    if irregular and not extrap:
        seq_len = sample_size
    if extrap:
        seq_len += 1

    netD_img = Discriminator(in_ch=cfg.MODEL.INPUT_DIM, device=device, seq=False,
                             is_extrap=extrap).to(device)
    netD_seq = Discriminator(in_ch=cfg.MODEL.INPUT_DIM * (seq_len), device=device, seq=True,
                             is_extrap=extrap).to(device)

    # Optimizer
    optimizer_netD = optim.Adamax(list(netD_img.parameters()) + list(netD_seq.parameters()), lr=lr)
    return netD_img, netD_seq, optimizer_netD