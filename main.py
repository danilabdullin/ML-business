'''
Модель сегментации инсульта головного мозга.
input: path to DICOM file(.gz or directory)
output: бинарная маска размера: слоиХ112х112 сегментации инсульта
'''



import napari
import torch.nn as nn

import nibabel as nib
import SimpleITK as sitk
import torch
import numpy as np
from torchvision import transforms

device = 'cpu'

# определенние нейронной сети, которое нужно для последующей загрузки весов
class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # Левая сторона (Путь уменьшения размерности картинки)
        self.down_conv_11 = self.conv_block(in_channels=1,
                                            out_channels=64)
        self.down_conv_12 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_21 = self.conv_block(in_channels=64,
                                            out_channels=128)
        self.down_conv_22 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_31 = self.conv_block(in_channels=128,
                                            out_channels=256)
        self.down_conv_32 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_41 = self.conv_block(in_channels=256,
                                            out_channels=512)
        self.down_conv_42 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)

        self.middle = self.conv_block(in_channels=512, out_channels=1024)

        # Правая сторона (Путь увеличения размерности картинки)
        self.up_conv_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_12 = self.conv_block(in_channels=1024,
                                          out_channels=512)
        self.up_conv_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_22 = self.conv_block(in_channels=512,
                                          out_channels=256)
        self.up_conv_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_32 = self.conv_block(in_channels=256,
                                          out_channels=128)
        self.up_conv_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_42 = self.conv_block(in_channels=128,
                                          out_channels=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes,
                                kernel_size=3, stride=1,
                                padding=1)

    #         self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def conv_block(in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels))
        return block

    @staticmethod
    def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, X):
        # Проход по левой стороне
        x1 = self.down_conv_11(X)  # [-1, 64, 256, 256]
        x2 = self.down_conv_12(x1)  # [-1, 64, 128, 128]
        x3 = self.down_conv_21(x2)  # [-1, 128, 128, 128]
        x4 = self.down_conv_22(x3)  # [-1, 128, 64, 64]
        x5 = self.down_conv_31(x4)  # [-1, 256, 64, 64]
        x6 = self.down_conv_32(x5)  # [-1, 256, 32, 32]
        x7 = self.down_conv_41(x6)  # [-1, 512, 32, 32]
        x8 = self.down_conv_42(x7)  # [-1, 512, 16, 16]

        middle_out = self.middle(x8)  # [-1, 1024, 16, 16]

        # Проход по правой стороне
        x = self.up_conv_11(middle_out)  # [-1, 512, 32, 32]
        y = self.crop_tensor(x, x7)
        x = self.up_conv_12(torch.cat((x, y), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]

        x = self.up_conv_21(x)  # [-1, 256, 64, 64]
        y = self.crop_tensor(x, x5)
        x = self.up_conv_22(torch.cat((x, y), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]

        x = self.up_conv_31(x)  # [-1, 128, 128, 128]
        y = self.crop_tensor(x, x3)
        x = self.up_conv_32(torch.cat((x, y), dim=1))  # [-1, 256, 128, 128] -> [-1, 128, 128, 128]

        x = self.up_conv_41(x)  # [-1, 64, 256, 256]
        y = self.crop_tensor(x, x1)
        x = self.up_conv_42(torch.cat((x, y), dim=1))  # [-1, 128, 256, 256] -> [-1, 64, 256, 256]

        output = self.output(x)  # [-1, num_classes, 256, 256]
        #         output = self.softmax(output)

        return output

# определение нейронной сети
Unet = UNet(num_classes=1)
# загрузка преодобученных весов, который доступны по ссылке:
# 'https://drive.google.com/file/d/1ZCVcWxuD88CUJF1TmH8MUZu_iREFo-hE/view?usp=sharing'
model = torch.load('best_model.pth')

# проерка, чтобы были рпавельные слеши
def make_path_right(path):
    path = path.replace('\\', '/')
    return path

# функция преобработки DICOM файла
def normalize_tensor(data):
    diff = torch.max(data) - torch.min(data)
    if diff == 0:
        return torch.zeros_like(data)
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

# функция преобработки DICOM файла
def preprocess(tensor):
    image = torch.Tensor(tensor)
    i = 0
    for s in image:
        image[i] = normalize_tensor(s)
        i += 1

    resize = transforms.Resize(size=(112, 112), antialias=None)
    image = resize(image)
    image = image.unsqueeze(1)
    shape_of_file = image.shape[0]

    return image

# загрузка и предобработка DICOM файла, который может быть как в формате папки, так и в формате файла .gz
def load_dicom(path):
    path = make_path_right(path)
    if path[-2:] == 'gz':
        image = nib.load(path).get_fdata()
    else:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image_itk = reader.Execute()

        image = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    image = preprocess(image)
    return image

# функция предсказания
def predict_segmentation(image):
    i = 0
    shape_of_file = image.shape[0]
    result = np.zeros((shape_of_file, 1, 112, 112))
    for m in image:
        m = m.unsqueeze(0)
        m = m.float()

        output = model(m.to(device))
        preds = torch.sigmoid(output) > 0.5
        result[i] = preds.detach().cpu().numpy()
        i += 1

    return result


# для работы нужно передать path c DICOM файлом пациента. На выходе получим маску сегментации для DICOM файла с размером
# (кол-во слоев,112,112), что значит, что для отрисовки 3D сегментации на мозге, нужно, чтобы DICOM файд так же был 112х112
path = 'subset/Brain_stroke_005'
model.to(device)
image = load_dicom(path)
result = predict_segmentation(image)


# отрисовка результата
# viewer = napari.Viewer()
# viewer.add_image(image.squeeze(1), name='DICOM')
# viewer.add_labels(result.squeeze(1).astype(int), name='Preds')
# napari.run()

