#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import cv2
import copy
import numpy as np

##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 41)  # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = False  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'bsds500'


# for strOption, strArgument in \
# getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
#     if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use
#     if strOption == '--in' and strArgument != '': arguments_strIn = strArgument  # path to the input image
#     if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
#

# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

    # end

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tensorInput = torch.cat([tensorBlue, tensorGreen, tensorRed], 1)

        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)

        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

        tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)
        tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv,
                                                         size=(tensorInput.size(2), tensorInput.size(3)),
                                                         mode='bilinear', align_corners=False)

        return self.moduleCombine(
            torch.cat([tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv], 1))
# end


# end

moduleNetwork = Network().eval()


##########################################################

def estimate(tensorInput):
    intWidth = tensorInput.size(2)
    intHeight = tensorInput.size(1)

    # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    return moduleNetwork(tensorInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


# end

##########################################################

if __name__ == '__main__':

    for count_number in range(38):
        arguments_strIn = "./small/"+str(count_number+1)+"_1.jpg"
        #arguments_strIn="./small/37_1.jpg"
        tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
        tensorOutput = estimate(tensorInput)

        ML_image = PIL.Image.fromarray(
            (tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8))

        img = cv2.cvtColor(numpy.asarray(ML_image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("./origin_output.jpg",img)
        # 这里从gray_iamge变成二值图像吧
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
        thresh_clone = copy.copy(thresh)

        cv2.imwrite("./thresh.jpg",thresh)
        #这里找好二值图片之后去找contour
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_index = 0
        for i in range(len(contours)):
            if(cv2.arcLength(contours[i],True)>cv2.arcLength(contours[max_index],True)):
                max_index = i

        #print(max_index)
        #到这里求出了最大的contour的下标，现在对这个contour求旋转矩形
        origin_iamge = cv2.imread(arguments_strIn)

        #debug drawcontous
        cv2.drawContours(origin_iamge,contours,max_index,(0,255,100),1)
        cv2.imwrite("./contour.jpg",origin_iamge)

        rotatedrect = cv2.minAreaRect(contours[max_index])
        # #debug 画出BoundingRect
        # box = cv2.boxPoints(rotatedrect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        # box = np.int0(box)
        # # 画出来
        # cv2.drawContours(origin_iamge, [box], 0, (255, 0, 0), 1)
        # cv2.imwrite('./boundingrect_iamga.jpg', origin_iamge)
        # #debug，这里是防止图像旋转之后超出范围
        # rotated_width = math.ceil(thresh_clone.rows * math.fabs(math.sin(rotatedrect[2] * 3.14 / 180)) + thresh_clone.cols * math.fabs(math.cos(rotatedrect[2] * 3.14 / 180)))
        # rotated_height = math.ceil(thresh_clone.cols * math.fabs(math.sin(rotatedrect[2] * 3.14 / 180)) + thresh_clone.rows * math.fabs(math.cos(rotatedrect[2] * 3.14 / 180)))
        # rotate_matrix = cv2.getRotationMatrix2D(rotatedrect[0], angle, 1.0)
        # rotate_matrix.at < double > (0, 2) += (rotated_width - src.cols) / 2
        # rotate_matrix.at < double > (1, 2) += (rotated_height - src.rows) / 2
        # #debug,这里有时间再来看
        transform_mat = cv2.getRotationMatrix2D(rotatedrect[0],rotatedrect[2],1.0)
        roctated_origin_iamge=cv2.warpAffine(origin_iamge,transform_mat,(origin_iamge.shape[0]*2,origin_iamge.shape[1]*2))
        rota_bin_image = cv2.warpAffine(thresh_clone, transform_mat, (origin_iamge.shape[0]*2, origin_iamge.shape[1]*2))
        cv2.imwrite("./roctated_origin_iamge.jpg",roctated_origin_iamge)
        cv2.imwrite("./rota_bin_image.jpg",rota_bin_image)
        #再对rota_bin_image找轮廓，找出boundingrect再找出roctated_origin_iamge的ＲＯＩ
        last_contour,last_hierachy  =cv2.findContours(rota_bin_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        max_index = 0
        for i in range(len(last_contour)):
            if (cv2.arcLength(last_contour[i],True) > cv2.arcLength(last_contour[max_index],True)):
                max_index = i

        POI_points  = cv2.boundingRect(last_contour[max_index])
        res = roctated_origin_iamge[POI_points[1]: POI_points[1] + POI_points[3],POI_points[0]:POI_points[0]+POI_points[2]]
        String_out = "./small/"+str(count_number+1)+"_2.jpg"
        cv2.imwrite(String_out,res)
        #cv2.imwrite("./String_out.jpg",res)


# end
# 我这里也不去参考java版本的实现了，直接来写一个简单的找轮廓裁切操作，再加上桌面的base64的操作，然后扔到flask上面
# 现在我有一个旋转矩形的中心和角度　这个矩形的宽和高，我甚至只要粗略地把input图片整张旋转，不行，不能把整张图片旋转，因为旋转角度跟原图的宽高有关系，
#　整张图片旋转会在旋转角度为负数的时候将宽变成高，将高变成宽，其实把图片扩出来无所谓，因为二值图像的补值对二值图像的找轮廓没有影响，最后出来的反正是割好的矩形，完美
# 那么我可以留着这个

#第二个思路是我用ｈｅｄ找到轮廓图之后，我对轮廓用houghlinesp找直线，我想要从这一堆直线中找出构成矩形的四条直线
#vector<四条直线组成的矩形>　遍历这个vector找出围成矩形面积最大的四条直线,再去求这四条直线的交点，求出交点之后
#求这四个点的旋转矩形，求仿射变换，将原图旋转正确，求这四个点旋转之后的坐标，求这四个点的boundingrect，将原图按照这个ｒｅｃｔ切出来

#(图片,极径分辨率通常是１越小越能检测小的直线,极角分辨率越小越能检测直线，所需最小交点越小直线就越多)
#lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 10, minLineLength=30, maxLineGap=5)

