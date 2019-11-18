"""
卷积的直观了解
参考链接:
https://blog.csdn.net/newchenxf/article/details/79375266
"""
# 参考代码：http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

# 二维卷积
def convolve2d(image, kernel):
    # This function which takes an image and a kernel(输入参数是一张图片和卷积核)
    # and returns the convolution of them(返回卷积的结果)
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    print('kernel = \n', kernel)
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel:将卷积核矩阵反转
    print('after flip, now kernel = \n', kernel)
    kernel_height = kernel.shape[0]  # 获得卷积核的高
    kernel_width = kernel.shape[1]  # 获得卷积核的宽

    output = np.zeros_like(image)  # convolution output:卷积输出(输出的大小和输入图像一致)
    print("image.shape:", image.shape, "output.shape:", output.shape)
    # Add zero padding to the input image
    # 左右上下要扩充一个像素, 否则边沿的像素，如（0，0）点，没法计算
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for width in range(image.shape[1]):     # Loop over every pixel of the image(根据图像的宽)
        for height in range(image.shape[0]):  # 根据图像的高
            # element-wise multiplication of the kernel and the image
            # 计算每一个卷积核下的卷积结果(需要求和)
            output[height, width] = \
                (kernel*image_padded[height:height+kernel_height, width:width+kernel_width]).sum()
    return output

# 卷积核反转
def reverse_cal(kernel):
    reverse = kernel.shape[0]
    kernel_fan = np.zeros_like(kernel)
    for i in range(reverse):
        kernel_fan[i] = kernel[reverse-i-1]
    return kernel_fan

# 一维卷积
def conv1d(information, kernel):
    print('information = \n', information)
    print('kernel = \n', kernel)
    kernel_fan = reverse_cal(kernel)
    print('after flip, now kernel = \n', kernel_fan)
    max_length = len(information) + len(kernel) - 1

    kernel_output = np.zeros((len(information), max_length))
    for i in range(len(information)):
        kernel_output[i, i:i+len(kernel)] = kernel
    output = np.dot(information, kernel_output)
    print(output)


"""
# 二维卷积调用
img = io.imread('F:\\test\\-41a8097ab730c82b.jpg')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel):RGB是3通道,这里转换成1通道

# Adjust the contrast of the image by applying Histogram Equalization
# 通过直方图均衡化调整图像的对比度
# image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
# plt.imshow(image_equalized, cmap='gray')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# Convolve the sharpen kernel and the image :卷积核作用:图像锐化
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

image_sharpen = convolve2d(img, kernel)
# 输出前5*5的
print('\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5, :5]*255)

# Plot the filtered image
plt.imshow(image_sharpen, cmap='gray')
plt.axis('off')
plt.show()
"""
# 一维卷积调用
information = np.array([1, 2, 3])
mykernel = np.array([2, 3, 1])
conv1d(information, mykernel)
