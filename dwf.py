import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utilities import prepare_input_data, get_cloud_mask_landsat_toa


class GatedConv2DActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.GELU()):
        super(GatedConv2DActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.InstanceNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation, stride, kernel_size):
        super().__init__()
        self.conv1 = GatedConv2DActivation(in_channels, out_channels, dilation=dilation, kernel_size=kernel_size,
                                           padding=padding)
        self.conv2 = GatedConv2DActivation(out_channels, out_channels, dilation=dilation, stride=stride,
                                           kernel_size=kernel_size, padding=padding)
        self.conv3 = GatedConv2DActivation(in_channels, out_channels, kernel_size=1)

    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        X = self.conv3(X)
        x += X
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1, add_last=False):
        super(ClassificationLayer, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(
            GatedResidualBlock(in_channels=in_channels, out_channels=out_channels, padding='same', dilation=1, stride=1,
                               kernel_size=5))
        for _ in range(repeat - 1):
            self.model.append(
                GatedResidualBlock(in_channels=out_channels, out_channels=out_channels, padding='same', dilation=1,
                                   stride=1,
                                   kernel_size=5))
        if add_last:
            self.out = nn.Conv2d(out_channels, 3, padding='same', dilation=1, stride=1, kernel_size=1)
            self.model.append(self.out)

    def forward(self, x):
        for block in self.model:
            x = block(x)
        return x


class RegressionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1, add_last=False):
        super(RegressionLayer, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(
            GatedResidualBlock(in_channels=in_channels, out_channels=out_channels, padding='same', dilation=1, stride=1,
                               kernel_size=5))
        for _ in range(repeat - 1):
            self.model.append(
                GatedResidualBlock(in_channels=out_channels, out_channels=out_channels, padding='same', dilation=1,
                                   stride=1,
                                   kernel_size=5))
        if add_last:
            self.out = nn.Conv2d(out_channels, 1, padding='same', dilation=1, stride=1, kernel_size=1)
            self.model.append(self.out)

    def forward(self, x):
        for block in self.model:
            x = block(x)
        return x


def create_upsample_layer(in_channels, out_channels):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=6, stride=2, padding=2),
        nn.GELU(),
        nn.InstanceNorm2d(out_channels)
    )
    return block


def create_max_pool_layer():
    block = nn.MaxPool2d(kernel_size=2, stride=2)
    return block


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class ClassificationModel(nn.Module):
    def __init__(self, base=1):
        super(ClassificationModel, self).__init__()
        repeat = 1
        self.keep_512_down = ClassificationLayer(10, base * 1, repeat=repeat)
        self.keep_256_down = ClassificationLayer(base * 1, base * 2, repeat=repeat)
        self.keep_128_down = ClassificationLayer(base * 2, base * 4, repeat=repeat)
        self.keep_64_down = ClassificationLayer(base * 4, base * 8, repeat=repeat * 2)
        self.keep_32_down = ClassificationLayer(base * 8, base * 16, repeat=repeat * 3)

        self.down_scale_2 = create_max_pool_layer()

        self.up_32_64 = create_upsample_layer(base * 16, base * 8)
        self.up_64_128 = create_upsample_layer(base * 16, base * 8)
        self.up_128_256 = create_upsample_layer(base * 12, base * 6)
        self.up_256_512 = create_upsample_layer(base * 8, base * 4)

        self.keep_64_up = ClassificationLayer(base * 8, base * 8, repeat=repeat * 2)
        self.keep_128_up = ClassificationLayer(base * 8, base * 8, repeat=repeat)
        self.keep_256_up = ClassificationLayer(base * 6, base * 6, repeat=repeat)

        self.keep_512_out = ClassificationLayer(base * 4, 3, repeat=repeat, add_last=True)

        self.device = 'cuda'

        self.apply(init_weights)

    def forward(self, sample):
        predictors = []
        for key in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'mndwi', 'lswi']:
            predictors.append(sample[key])
        x = torch.cat(predictors, dim=1)

        # down
        x_512_after_keep = self.keep_512_down(x)
        x_512_256 = self.down_scale_2(x_512_after_keep)
        x_256_after_keep = self.keep_256_down(x_512_256)
        x_256_128 = self.down_scale_2(x_256_after_keep)
        x_128_after_keep = self.keep_128_down(x_256_128)
        x_128_64 = self.down_scale_2(x_128_after_keep)
        x_64_after_keep = self.keep_64_down(x_128_64)
        x_64_32 = self.down_scale_2(x_64_after_keep)
        x_32_after_keep = self.keep_32_down(x_64_32)

        # up
        x_32_64 = self.up_32_64(x_32_after_keep)
        x_64_after_keep_up = self.keep_64_up(x_32_64)

        x_64_fused = torch.cat([x_64_after_keep_up, x_64_after_keep], dim=1)
        x_64_128 = self.up_64_128(x_64_fused)
        x_128_after_keep_up = self.keep_128_up(x_64_128)

        x_128_fused = torch.cat([x_128_after_keep_up, x_128_after_keep], dim=1)
        x_128_256 = self.up_128_256(x_128_fused)
        x_256_after_keep_up = self.keep_256_up(x_128_256)

        x_256_fused = torch.cat([x_256_after_keep_up, x_256_after_keep], dim=1)
        x_256_512 = self.up_256_512(x_256_fused)

        # out
        x_512_out = self.keep_512_out(x_256_512)

        pred = x_512_out
        return pred


cla_weight = "./pre-trained/cla.pt"
use_cla = False  # do not use classification model for speed up
if use_cla:
    model_cla = ClassificationModel(base=60)
    model_cla = model_cla.to('cuda')
    checkpoint = torch.load(cla_weight, map_location='cuda')
    model_cla.load_state_dict(checkpoint['model_state_dict'])
    model_cla.eval()


def dwf_prediction(blue, green, red, nir, swir1, swir2, qa):
    cloud_mask, snow_mask = get_cloud_mask_landsat_toa(qa, flags=[1, 3, 4], buffer=0)

    nan_mask = np.zeros_like(blue) > 1
    for band in [blue, green, red, nir, swir1, swir2]:
        band[band == -float('inf')] = 0
        band[cloud_mask > 0] = 0
        nan_mask |= (band == 0)

    x_size, y_size = blue.shape

    sample = prepare_input_data(blue, green, red, nir, swir1, swir2)

    x_offs = [0, 62, x_size - 64]
    y_offs = [0, 62, y_size - 64]

    x_off = 62
    while not x_off + 62 > x_size - 64:
        x_off += 62
        x_offs.append(x_off)

    y_off = 62
    while not y_off + 62 > y_size - 64:
        y_off += 62
        y_offs.append(y_off)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        dummy_pred_reg = np.zeros_like(blue)
        if use_cla:
            dummy_pred_cla = np.zeros_like(blue)
        for x_off in tqdm(x_offs):
            for y_off in y_offs:
                sample_ = {}
                for key in sample:
                    sample_[key] = sample[key][:, :, x_off:x_off + 64, y_off:y_off + 64]

                if x_off == 0:
                    x_start = 0
                else:
                    x_start = 1
                if y_off == 0:
                    y_start = 0
                else:
                    y_start = 1

                if x_off == x_size - 64:
                    x_end = 64
                else:
                    x_end = 63
                if y_off == y_size - 64:
                    y_end = 64
                else:
                    y_end = 63

                fraction_pred = model_reg(sample_)
                fraction_pred = np.squeeze(torch.sigmoid(fraction_pred)).cpu().detach().numpy()
                dummy_pred_reg[x_off + x_start:x_off + x_end, y_off + y_start:y_off + y_end] = fraction_pred[
                                                                                               x_start:x_end,
                                                                                               y_start:y_end]

                if use_cla:
                    if (fraction_pred[1:-1, 1:-1] > 0.9).all():
                        dummy_pred_cla[x_off + x_start:x_off + x_end, y_off + y_start:y_off + y_end] = 2
                        continue
                    if (fraction_pred[1:-1, 1:-1] < 0.1).all():
                        dummy_pred_cla[x_off + x_start:x_off + x_end, y_off + y_start:y_off + y_end] = 0
                        continue
                    cla_pred = model_cla(sample_)
                    cla_pred = np.squeeze(torch.argmax(cla_pred, dim=1)).cpu().detach().numpy()
                    dummy_pred_cla[x_off + x_start:x_off + x_end, y_off + y_start:y_off + y_end] = cla_pred[
                                                                                                   x_start:x_end,
                                                                                                   y_start:y_end]

    if use_cla:
        dummy_pred_cla[nan_mask] = np.nan
        dummy_pred_reg[(dummy_pred_reg > 0.95) & (dummy_pred_cla != 1)] = 1
        dummy_pred_reg[(dummy_pred_reg < 0.05) & (dummy_pred_cla != 1)] = 0
    else:
        dummy_pred_reg[dummy_pred_reg > 0.95] = 1
        dummy_pred_reg[dummy_pred_reg < 0.05] = 0

    dummy_pred_reg[nan_mask] = np.nan
    return dummy_pred_reg


class RegressionModel(nn.Module):
    def __init__(self, base=1):
        super(RegressionModel, self).__init__()
        repeat = 1
        self.keep_512_down = RegressionLayer(10, base * 1, repeat=repeat)
        self.keep_256_down = RegressionLayer(base * 1, base * 2, repeat=repeat)
        self.keep_128_down = RegressionLayer(base * 2, base * 4, repeat=repeat)
        self.keep_64_down = RegressionLayer(base * 4, base * 8, repeat=repeat * 2)
        self.keep_32_down = RegressionLayer(base * 8, base * 16, repeat=repeat * 3)

        self.down_scale_2 = create_max_pool_layer()

        self.up_32_64 = create_upsample_layer(base * 16, base * 8)
        self.up_64_128 = create_upsample_layer(base * 16, base * 8)
        self.up_128_256 = create_upsample_layer(base * 12, base * 6)
        self.up_256_512 = create_upsample_layer(base * 8, base * 4)

        self.keep_64_up = RegressionLayer(base * 8, base * 8, repeat=repeat * 2)
        self.keep_128_up = RegressionLayer(base * 8, base * 8, repeat=repeat)
        self.keep_256_up = RegressionLayer(base * 6, base * 6, repeat=repeat)

        self.keep_64_out = RegressionLayer(base * 8, 1, repeat=repeat, add_last=True)
        self.keep_128_out = RegressionLayer(base * 8, 1, repeat=repeat, add_last=True)
        self.keep_256_out = RegressionLayer(base * 6, 1, repeat=repeat, add_last=True)
        self.keep_512_out = RegressionLayer(base * 4, 1, repeat=repeat, add_last=True)

        self.device = 'cuda'

    def forward(self, sample):
        predictors = []
        for key in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'mndwi', 'lswi']:
            predictors.append(sample[key])
        x = torch.cat(predictors, dim=1)

        # down
        x_512_after_keep = self.keep_512_down(x)
        x_512_256 = self.down_scale_2(x_512_after_keep)
        x_256_after_keep = self.keep_256_down(x_512_256)
        x_256_128 = self.down_scale_2(x_256_after_keep)
        x_128_after_keep = self.keep_128_down(x_256_128)
        x_128_64 = self.down_scale_2(x_128_after_keep)
        x_64_after_keep = self.keep_64_down(x_128_64)
        x_64_32 = self.down_scale_2(x_64_after_keep)
        x_32_after_keep = self.keep_32_down(x_64_32)

        # up
        x_32_64 = self.up_32_64(x_32_after_keep)
        x_64_after_keep_up = self.keep_64_up(x_32_64)

        x_64_fused = torch.cat([x_64_after_keep_up, x_64_after_keep], dim=1)
        x_64_128 = self.up_64_128(x_64_fused)
        x_128_after_keep_up = self.keep_128_up(x_64_128)

        x_128_fused = torch.cat([x_128_after_keep_up, x_128_after_keep], dim=1)
        x_128_256 = self.up_128_256(x_128_fused)
        x_256_after_keep_up = self.keep_256_up(x_128_256)

        x_256_fused = torch.cat([x_256_after_keep_up, x_256_after_keep], dim=1)
        x_256_512 = self.up_256_512(x_256_fused)

        # out
        x_512_out = self.keep_512_out(x_256_512)

        pred = x_512_out
        return pred


print('loading model')
regression_weight = "./pre-trained/reg.pt"
model_reg = RegressionModel(base=60)
model_reg = model_reg.to('cuda')
checkpoint = torch.load(regression_weight, map_location='cuda')
model_reg.load_state_dict(checkpoint['model_state_dict'])
model_reg.eval()
print('model loaded')
