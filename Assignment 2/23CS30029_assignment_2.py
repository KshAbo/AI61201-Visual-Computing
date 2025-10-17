#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# In[57]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
directory = "./inputs/VisualComputingAssignment2_inputs/"


# # Task 1: Image Exposure Check

# In[ ]:


# The function classifies whether an image is underexposed, overexposed or correctly exposed based on the mean brightness of the grayscale version of the image
def checkExposure(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean_brightness = img_gray.mean()

    if(mean_brightness < 80):
        return "underexposed"
    elif(mean_brightness > 180):
        return "overexposed"
    else:
        return "correctly exposed"

VC_a = cv2.imread(directory+"VC_a.jpg")
VC_a_gray = cv2.cvtColor(VC_a, cv2.COLOR_BGR2RGB)
VC_b = cv2.imread(directory+"VC_b.JPG")
VC_b_gray = cv2.cvtColor(VC_b, cv2.COLOR_BGR2RGB)
VC_c = cv2.imread(directory+"VC_c.jpg")
VC_c_gray = cv2.cvtColor(VC_c, cv2.COLOR_BGR2RGB)
VC_d = cv2.imread(directory+"VC_d.jpg")
VC_d_gray = cv2.cvtColor(VC_d, cv2.COLOR_BGR2RGB)
VC_e = cv2.imread(directory+"VC_e.jpg")
VC_e_gray = cv2.cvtColor(VC_e, cv2.COLOR_BGR2RGB)
VC_f = cv2.imread(directory+"VC_f.jpg")
VC_f_gray = cv2.cvtColor(VC_f, cv2.COLOR_BGR2RGB)

print("Image VC_a is " + checkExposure(VC_a))
print("Image VC_b is " + checkExposure(VC_b))
print("Image VC_c is " + checkExposure(VC_c))
print("Image VC_d is " + checkExposure(VC_d))
print("Image VC_e is " + checkExposure(VC_e))
print("Image VC_f is " + checkExposure(VC_f))


# ## Task 2: Separable Gaussian Filtering

# In[ ]:


import timeit

noisy_image = cv2.imread(directory + "noisy_image.jpg")
noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

# Generates a 1D Gaussian kernel
def gaussian_1d(size, sigma):
    kernel = np.arange((-size//2), size//2 + 1)
    kernel = np.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

separable_gaussian_kernel = gaussian_1d(3, 11)

constant = np.matmul(separable_gaussian_kernel.T.reshape(-1, 1), separable_gaussian_kernel.reshape(1, -1)).sum()


def convolution(img, kernel):

    img_convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 
                                                axis=0, 
                                                arr=img)
    img_convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 
                                                axis=1, 
                                                arr=img_convolved)
    return img_convolved


start = timeit.default_timer()
noisy_img_convolved = convolution(noisy_image, separable_gaussian_kernel)
end = timeit.default_timer()
separable_time = end - start
print(f"Time taken for convolution using separable filter: {end - start} seconds")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].set_title("Noisy Image")
axes[0].imshow(noisy_image, cmap='gray')
axes[0].axis('off')
axes[1].set_title("Denoised Image")
axes[1].imshow(noisy_img_convolved, cmap='gray')
axes[1].axis('off')
plt.show()


# ## Tutorial implementation of Gaussian Filter

# In[60]:


def gaussian_filter(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            -((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def convolution2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)), mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

start = timeit.default_timer()

noisy_image_convolved_tut = convolution2d(noisy_image, gaussian_filter(3, 11))

end = timeit.default_timer()

tutorial_time = end - start
print(f"Time taken for convolution using 2D filter: {end - start} seconds")

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
axes[0].set_title("Noisy Image")
axes[0].imshow(noisy_image, cmap='gray')
axes[0].axis('off')
axes[1].set_title("Denoised Image from tutorial")
axes[1].imshow(noisy_image_convolved_tut, cmap='gray')
axes[1].axis('off')
axes[2].set_title("Denoised Image from separable filter")
axes[2].imshow(noisy_img_convolved, cmap='gray')
axes[2].axis('off')
plt.show()


# ### The image output from both 2D convolution and seperable filter look the same

# #### Finding the speed difference between using a separable kernel and a matrix

# In[61]:


# Comparing the times
print(f"Time taken using separable filter: {separable_time} seconds")
print(f"Time taken using 2D filter: {tutorial_time} seconds")
print(f"The speedup factor is: {tutorial_time / separable_time}")


# ## Task 3: Texture Classification Using Gabor Filtering

# ### 3.a

# In[ ]:


# Defining a function that returns a Gabor filter
def gaborFilter(size=50, sigma=5, gamma=1, u=0.05, v=0.05, phi=0):
    x, y = np.meshgrid(np.linspace(-size//2, size//2, size), np.linspace(-size//2, size//2, size))

    gb = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            gb[i, j] = np.exp(- (x[i, j]**2 + (gamma**2) * y[i, j]**2) / (2 * sigma**2)) * np.cos(2 * np.pi * (u * x[i, j] + v * y[i, j]) + phi)

    return gb

gb_0deg = gaborFilter(u=0.05, v=0)
gb_45deg = gaborFilter(u=0.05, v=0.05)
gb_90deg = gaborFilter(u=0, v=0.05)
gb_135deg = gaborFilter(u=-0.05, v=0.05)

gb_dot25freq = gaborFilter(u=0.25, v=0)
gb_dot125freq = gaborFilter(u=0.125, v=0)
gb_dot0625freq = gaborFilter(u=0.0625, v=0)

gb_sigma_2 = gaborFilter(sigma=2)
gb_sigma_4 = gaborFilter(sigma=4)
gb_sigma_8 = gaborFilter(sigma=8)

gb_gamma_dot5 = gaborFilter(gamma=0.5)
gb_gamma_1 = gaborFilter(sigma=1)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes[0, 0].set_title("Gabor Filter 0 degrees")
axes[0, 0].imshow(gb_0deg, cmap='gray')
axes[0, 0].axis('off')
axes[0, 1].set_title("Gabor Filter 45 degrees")
axes[0, 1].imshow(gb_45deg, cmap='gray')
axes[0, 1].axis('off')
axes[0, 2].set_title("Gabor Filter 90 degrees")
axes[0, 2].imshow(gb_90deg, cmap='gray')
axes[0, 2].axis('off')
axes[0, 3].set_title("Gabor Filter 135 degrees")
axes[0, 3].imshow(gb_135deg, cmap='gray')
axes[0, 3].axis('off')
axes[1, 0].set_title("Gabor Filter 0.25 frequency")
axes[1, 0].imshow(gb_dot25freq, cmap='gray')
axes[1, 0].axis('off')
axes[1, 1].set_title("Gabor Filter 0.125 frequency")
axes[1, 1].imshow(gb_dot125freq, cmap='gray')
axes[1, 1].axis('off')
axes[1, 2].set_title("Gabor Filter 0.0625 frequency")
axes[1, 2].imshow(gb_dot0625freq, cmap='gray')
axes[1, 2].axis('off')
axes[1, 3].set_title("Gabor Filter sigma=2")
axes[1, 3].imshow(gb_sigma_2, cmap='gray')
axes[1, 3].axis('off')
axes[2, 0].set_title("Gabor Filter sigma=4")
axes[2, 0].imshow(gb_sigma_4, cmap='gray')
axes[2, 0].axis('off')
axes[2, 1].set_title("Gabor Filter sigma=8")
axes[2, 1].imshow(gb_sigma_8, cmap='gray')
axes[2, 1].axis('off')
axes[2, 2].set_title("Gabor Filter gamma=0.5")
axes[2, 2].imshow(gb_gamma_dot5, cmap='gray')
axes[2, 2].axis('off')
axes[2, 3].set_title("Gabor Filter gamma=1")
axes[2, 3].imshow(gb_gamma_1, cmap='gray')
axes[2, 3].axis('off')
plt.show()


# ### 3.b

# In[63]:


import skimage as ski

brick_img = ski.data.brick()
gravel_img = ski.data.gravel()
grass_img = ski.data.grass()

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].set_title("Brick Image")
axes[0].imshow(brick_img, cmap='gray')
axes[0].axis('off')
axes[1].set_title("Gravel Image")
axes[1].imshow(gravel_img, cmap='gray')
axes[1].axis('off')
axes[2].set_title("Grass Image")
axes[2].imshow(grass_img, cmap='gray')
axes[2].axis('off')
plt.show()


# In[ ]:


# Gabor filter bank with 3 frequencies and 4 orientations
gb_0deg_dot25freq       = gaborFilter(size=10, u=0.25, v=0)
gb_45deg_dot25freq      = gaborFilter(size=10, u=0.25, v=0.25)
gb_90deg_dot25freq      = gaborFilter(size=10, u=0, v=0.25)
gb_135deg_dot25freq     = gaborFilter(size=10, u=-0.25, v=0.25)
gb_0deg_dot125freq      = gaborFilter(size=10, u=0.125, v=0)
gb_45deg_dot125freq     = gaborFilter(size=10, u=0.125, v=0.125)
gb_90deg_dot125freq     = gaborFilter(size=10, u=0, v=0.125)
gb_135deg_dot25freq     = gaborFilter(size=10, u=-0.25, v=0.25)
gb_0deg_dot0625freq     = gaborFilter(size=10, u=0.0625, v=0)
gb_45deg_dot0625freq    = gaborFilter(size=10, u=0.0625, v=0.0625)
gb_90deg_dot0625freq    = gaborFilter(size=10, u=0, v=0.0625)
gb_135deg_dot0625freq   = gaborFilter(size=10, u=-0.0625, v=0.0625)

filter_bank = [gb_0deg_dot25freq, gb_45deg_dot25freq, gb_90deg_dot25freq, gb_135deg_dot25freq,
               gb_0deg_dot125freq, gb_45deg_dot125freq, gb_90deg_dot125freq, gb_135deg_dot25freq,
               gb_0deg_dot0625freq, gb_45deg_dot0625freq, gb_90deg_dot0625freq, gb_135deg_dot0625freq]


# In[ ]:


fig, axes = plt.subplots(13, 5, figsize=(50, 130))

axes[0, 0].axis('off')
axes[0, 1].axis('off')
axes[0, 2].set_title("Brick")
axes[0, 2].imshow(brick_img, cmap='gray')
axes[0, 2].axis('off')
axes[0, 3].set_title("Gravel")
axes[0, 3].imshow(gravel_img, cmap='gray')
axes[0, 3].axis('off')
axes[0, 4].set_title("Grass")
axes[0, 4].imshow(grass_img, cmap='gray')
axes[0, 4].axis('off')

# The number of features is 2 times the number of filters (mean and variance for each filter)
brick_feature  = np.zeros(filter_bank.__len__()*2)
gravel_feature = np.zeros(filter_bank.__len__()*2)
grass_feature  = np.zeros(filter_bank.__len__()*2)

# Applying the gabor filter for each image and calculating the mean and variance for later classification
for i, gb in enumerate(filter_bank):

    brick_filtered          = convolution2d(brick_img, gb)
    brick_feature[i]        = np.mean(brick_filtered)
    brick_feature[i + filter_bank.__len__()]  = np.var(brick_filtered)

    gravel_filtered         = convolution2d(gravel_img, gb)
    gravel_feature[i]       = np.mean(gravel_filtered)
    gravel_feature[i + filter_bank.__len__()] = np.var(gravel_filtered)

    grass_filtered          = convolution2d(grass_img, gb)
    grass_feature[i]        = np.mean(grass_filtered)
    grass_feature[i + filter_bank.__len__()]  = np.var(grass_filtered)

    axes[i+1, 0].set_title("Gabor Filter")
    axes[i+1, 0].imshow(gb, cmap='gray')
    axes[i+1, 0].axis('off')
    axes[i+1, 1].set_title("Level Set")
    axes[i+1, 1].contour(gb, levels=5, colors='black')
    axes[i+1, 1].axis('off')
    axes[i+1, 2].set_title(f"Brick Filtered {i+1}")
    axes[i+1, 2].imshow(brick_filtered, cmap='gray')
    axes[i+1, 2].axis('off')
    axes[i+1, 3].set_title(f"Gravel Filtered {i+1}")
    axes[i+1, 3].imshow(gravel_filtered, cmap='gray')
    axes[i+1, 3].axis('off')
    axes[i+1, 4].set_title(f"Grass Filtered {i+1}")
    axes[i+1, 4].imshow(grass_filtered, cmap='gray')
    axes[i+1, 4].axis('off')

plt.show()


# In[ ]:


img1 = cv2.imread(directory + "img1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(directory + "img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread(directory + "img3.jpg")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.imread(directory + "img4.jpg")
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5 = cv2.imread(directory + "img5.jpg")
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
img6 = cv2.imread(directory + "img6.jpg")
img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

imgi_features = np.zeros((6, filter_bank.__len__()*2))

# Applying the gabor filter for each image and calculating the mean and variance for later classification
for i, gb in enumerate(filter_bank):

    imgi                            = convolution2d(img1, gb)
    imgi_features[0][i]             = np.mean(imgi)
    imgi_features[0][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img2, gb)
    imgi_features[1][i]             = np.mean(imgi)
    imgi_features[1][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img3, gb)
    imgi_features[2][i]             = np.mean(imgi)
    imgi_features[2][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img4, gb)
    imgi_features[3][i]             = np.mean(imgi)
    imgi_features[3][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img5, gb)
    imgi_features[4][i]             = np.mean(imgi)
    imgi_features[4][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(imgi, gb)
    imgi_features[5][i]             = np.mean(imgi)
    imgi_features[5][i + filter_bank.__len__()]       = np.var(imgi)


# In[67]:


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
data = pd.DataFrame([brick_feature])
data = pd.concat([data, pd.DataFrame([gravel_feature])])
data = pd.concat([data, pd.DataFrame([grass_feature])])
data = pd.concat([data, pd.DataFrame(imgi_features)], axis=0)
data.columns = [f"feature{i+1}" for i in range(data.shape[1])]
data.index = ["Brick", "Gravel", "Grass", "img1", "img2", "img3", "img4", "img5", "img6"]
data1 = data
print(data)


# #### Calculating the absolute differnce of each corresponding feature with the features of Brick, Grass and Gravel and choosing the minimum to classify them

# In[ ]:


for i in range(3, 9):

    # Distance caculation of each class
    dist_brick   = np.linalg.norm(data.iloc[0] - data.iloc[i])
    dist_gravel  = np.linalg.norm(data.iloc[1] - data.iloc[i])
    dist_grass   = np.linalg.norm(data.iloc[2] - data.iloc[i])

    # Storing the distances and the textures respectively
    distances = [dist_brick, dist_gravel, dist_grass]
    textures = ["Brick", "Gravel", "Grass"]

    # Choosing the minimum distance as the predicted texture
    texture_pred = textures[np.argmin(distances)]

    print(f"The predicted texture for img{i-2} is {texture_pred} with distances (Brick: {dist_brick}, Gravel: {dist_gravel}, Grass: {dist_grass})")


# #### Features from just mean and variance from the Gabor features computed cannot really classify whether an image is Brick, Grass or Gravel

# ### 3.c

# In[69]:


gb_0deg_dot25freq_cv       = cv2.getGaborKernel(ksize=(10, 10), theta=0,          lambd=1/0.25, sigma=5, gamma=1, psi=0)
gb_45deg_dot25freq_cv      = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/4,    lambd=1/0.25, sigma=5, gamma=1, psi=0)
gb_90deg_dot25freq_cv      = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/2,    lambd=1/0.25, sigma=5, gamma=1, psi=0)
gb_135deg_dot25freq_cv     = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi*3/4,  lambd=1/0.25, sigma=5, gamma=1, psi=0)
gb_0deg_dot125freq_cv      = cv2.getGaborKernel(ksize=(10, 10), theta=0,          lambd=1/0.125, sigma=5, gamma=1, psi=0)
gb_45deg_dot125freq_cv     = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/4,    lambd=1/0.125, sigma=5, gamma=1, psi=0)
gb_90deg_dot125freq_cv     = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/2,    lambd=1/0.125, sigma=5, gamma=1, psi=0)
gb_135deg_dot25freq_cv     = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi*3/4,  lambd=1/0.25, sigma=5, gamma=1, psi=0)
gb_0deg_dot0625freq_cv     = cv2.getGaborKernel(ksize=(10, 10), theta=0,          lambd=1/0.0625, sigma=5, gamma=1, psi=0)
gb_45deg_dot0625freq_cv    = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/4,    lambd=1/0.0625, sigma=5, gamma=1, psi=0)
gb_90deg_dot0625freq_cv    = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi/2,    lambd=1/0.0625, sigma=5, gamma=1, psi=0)
gb_135deg_dot0625freq_cv   = cv2.getGaborKernel(ksize=(10, 10), theta=np.pi*3/4,  lambd=1/0.0625, sigma=5, gamma=1, psi=0)

filter_bank = [gb_0deg_dot25freq_cv, gb_45deg_dot25freq_cv, gb_90deg_dot25freq_cv, gb_135deg_dot25freq_cv,
               gb_0deg_dot125freq_cv, gb_45deg_dot125freq_cv, gb_90deg_dot125freq_cv, gb_135deg_dot25freq_cv,
               gb_0deg_dot0625freq_cv, gb_45deg_dot0625freq_cv, gb_90deg_dot0625freq_cv, gb_135deg_dot0625freq_cv]


# In[70]:


fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes[0, 0].set_title("Gabor Filter 0 degrees")
axes[0, 0].imshow(gb_0deg_dot25freq_cv, cmap='gray')
axes[0, 0].axis('off')
axes[0, 1].set_title("Gabor Filter 45 degrees")
axes[0, 1].imshow(gb_45deg_dot25freq_cv, cmap='gray')
axes[0, 1].axis('off')
axes[0, 2].set_title("Gabor Filter 90 degrees")
axes[0, 2].imshow(gb_90deg_dot25freq_cv, cmap='gray')
axes[0, 2].axis('off')
axes[0, 3].set_title("Gabor Filter 135 degrees")
axes[0, 3].imshow(gb_135deg_dot25freq_cv, cmap='gray')
axes[0, 3].axis('off')
axes[1, 0].set_title("Gabor Filter 0.125 frequency")
axes[1, 0].imshow(gb_0deg_dot125freq_cv, cmap='gray')
axes[1, 0].axis('off')
axes[1, 1].set_title("Gabor Filter 45 degrees")
axes[1, 1].imshow(gb_45deg_dot125freq_cv, cmap='gray')
axes[1, 1].axis('off')
axes[1, 2].set_title("Gabor Filter 90 degrees")
axes[1, 2].imshow(gb_90deg_dot125freq_cv, cmap='gray')
axes[1, 2].axis('off')
axes[1, 3].set_title("Gabor Filter 135 degrees")
axes[1, 3].imshow(gb_135deg_dot25freq_cv, cmap='gray')
axes[1, 3].axis('off')
axes[2, 0].set_title("Gabor Filter 0.0625 frequency")
axes[2, 0].imshow(gb_0deg_dot0625freq_cv, cmap='gray')
axes[2, 0].axis('off')
axes[2, 1].set_title("Gabor Filter 45 degrees")
axes[2, 1].imshow(gb_45deg_dot0625freq_cv, cmap='gray')
axes[2, 1].axis('off')
axes[2, 2].set_title("Gabor Filter 90 degrees")
axes[2, 2].imshow(gb_90deg_dot0625freq_cv, cmap='gray')
axes[2, 2].axis('off')
axes[2, 3].set_title("Gabor Filter 135 degrees")
axes[2, 3].imshow(gb_135deg_dot0625freq_cv, cmap='gray')
axes[2, 3].axis('off')
plt.show()


# In[71]:


brick_feature  = np.zeros(filter_bank.__len__()*2)
gravel_feature = np.zeros(filter_bank.__len__()*2)
grass_feature  = np.zeros(filter_bank.__len__()*2)

for i, gb in enumerate(filter_bank):

    brick_filtered          = convolution2d(brick_img, gb)
    brick_feature[i]        = np.mean(brick_filtered)
    brick_feature[i + filter_bank.__len__()]  = np.var(brick_filtered)

    gravel_filtered         = convolution2d(gravel_img, gb)
    gravel_feature[i]       = np.mean(gravel_filtered)
    gravel_feature[i + filter_bank.__len__()] = np.var(gravel_filtered)

    grass_filtered          = convolution2d(grass_img, gb)
    grass_feature[i]        = np.mean(grass_filtered)
    grass_feature[i + filter_bank.__len__()]  = np.var(grass_filtered)


# In[72]:


for i, gb in enumerate(filter_bank):

    imgi                            = convolution2d(img1, gb)
    imgi_features[0][i]             = np.mean(imgi)
    imgi_features[0][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img2, gb)
    imgi_features[1][i]             = np.mean(imgi)
    imgi_features[1][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img3, gb)
    imgi_features[2][i]             = np.mean(imgi)
    imgi_features[2][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img4, gb)
    imgi_features[3][i]             = np.mean(imgi)
    imgi_features[3][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(img5, gb)
    imgi_features[4][i]             = np.mean(imgi)
    imgi_features[4][i + filter_bank.__len__()]       = np.var(imgi)

    imgi                            = convolution2d(imgi, gb)
    imgi_features[5][i]             = np.mean(imgi)
    imgi_features[5][i + filter_bank.__len__()]       = np.var(imgi)


# In[82]:


pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
data = pd.DataFrame([brick_feature])
data = pd.concat([data, pd.DataFrame([gravel_feature])])
data = pd.concat([data, pd.DataFrame([grass_feature])])
data = pd.concat([data, pd.DataFrame(imgi_features)], axis=0)
data.columns = [f"feature{i+1}" for i in range(data.shape[1])]
data.index = ["Brick", "Gravel", "Grass", "img1", "img2", "img3", "img4", "img5", "img6"]
print(data)


# In[74]:


for i in range(3, 9):
    dist_brick   = np.linalg.norm(data.iloc[0] - data.iloc[i])
    dist_gravel  = np.linalg.norm(data.iloc[1] - data.iloc[i])
    dist_grass   = np.linalg.norm(data.iloc[2] - data.iloc[i])

    distances = [dist_brick, dist_gravel, dist_grass]
    textures = ["Brick", "Gravel", "Grass"]

    texture_pred = textures[np.argmin(distances)]

    print(f"The predicted texture for img{i-2} is {texture_pred} with distances (Brick: {dist_brick}, Gravel: {dist_gravel}, Grass: {dist_grass})")


# ## Task 4: Design of Sobel Operator 

# In[ ]:


tower_img = cv2.imread(directory + "tower.jpg")
tower_img_gray = cv2.cvtColor(tower_img, cv2.COLOR_BGR2GRAY)

# Sobel filter kernel, the filter gradient goes in the direction from bottom-left to top-right which should detect edges going in bottom-right to top-left direction
sobel_filter = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]])

tower_sobel = convolution2d(tower_img_gray, sobel_filter)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].set_title("Original Tower Image")
axes[0].imshow(tower_img_gray, cmap='gray')
axes[0].axis('off')
axes[1].set_title("Sobel Filtered Tower Image")
axes[1].imshow(tower_sobel, cmap='gray')
axes[1].axis('off')
plt.show()

