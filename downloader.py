import urllib.request

#url_10 = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-python.tar.gz"
#save_path_10 = "./data/cifar-10-batches-py.tar.gz"
url_100 = "https://mirrors.ustc.edu.cn/pytorch/cifar-100-python.tar.gz"
save_path_100 = "./data/cifar-100-batches-py.tar.gz"

#urllib.request.urlretrieve(url_10, save_path_10)
#print("CIFAR-10 dataset downloaded successfully.")
urllib.request.urlretrieve(url_100, save_path_100)
print("CIFAR-100 dataset downloaded successfully.")


