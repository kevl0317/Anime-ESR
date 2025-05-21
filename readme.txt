experiment —— 训练文件,置于项目根目录
	pretrained为用于step1训练的原始模型
	RealESRNET为Anime数据集训练的step1模型,对应options/train_realesrnet_x4plus.yml
	RealESRGAN为Anime数据集训练的step2模型，对应options/train_realesrgan_x4plus.yml
	RealESRGAN_Sobel为加入sobel loss和TV*Sobel loss的step2模型。对应options/train_realesrgan_sobel_x4plus.yml

dataset —— dataset文件,置于项目根目录
生成过程follow doc/train.md

datasetPrep ——下载pixiv图片，处理anime图片
下载pixiv图片参考DownloadByUser.py， 需提前按教程提取token，见PixivAuth.py和pixivpy的readme