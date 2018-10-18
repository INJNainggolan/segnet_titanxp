import sys

path = "/home/dpakhom1/dense_crf_python/"
sys.path.append(path)

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

import skimage.io as io

image = train_image

softmax = final_probabilities.squeeze()

softmax = processed_probabilities.transpose((2, 0, 1))

# 输入数据应为概率值的负对数
# 你可以在softmax_to_unary函数的定义中找到更多信息
unary = softmax_to_unary(processed_probabilities)

# 输入数据应为C-连续的——我们使用了Cython封装器
unary = np.ascontiguousarray(unary)

d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

d.setUnaryEnergy(unary)

# 潜在地对空间上相邻的小块分割区域进行惩罚——促使产生更多空间连续的分割区域
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

d.addPairwiseEnergy(feats, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

# 这将创建与颜色相关的图像特征——因为我们从卷积神经网络中得到的分割结果非常粗糙，
# 我们可以使用局部的颜色特征来改善分割结果
feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                  img=image, chdim=2)

d.addPairwiseEnergy(feats, compat=10,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
Q = d.inference(5)

res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

cmap = plt.get_cmap('bwr')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
ax1.set_title('Segmentation with CRF post-processing')
probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
ax2.set_title('Ground-Truth Annotation')
plt.show()