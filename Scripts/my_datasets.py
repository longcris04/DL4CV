# from torchvision.datasets import CIFAR10, ImageFolder

# dataset = CIFAR10(root="my_dataset", train=True, download=False)
# dataset = ImageFolder(root="./../Datasets/animals")

# image,label = dataset[1000]
# print(label)
# image.show()
# print(dataset.classes)

from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import CIFAR10
import pickle
import os
import numpy as np
import cv2

class MyCifar10(Dataset):
    def __init__(self, path, train):
        self.list_labels = []
        self.list_images = []
        if train:
            data_files = [os.path.join(path,f"data_batch_{i}") for i in range(1,6)]
        else:
            data_files = [os.path.join(path,"test_batch")]

        for data_file in data_files:
            with open(data_file,"rb") as file_:
                data = pickle.load(file_,encoding="bytes")
                images = data[b"data"]
                self.list_images.extend(images)
                
                self.list_labels.extend(data[b"labels"])
                

        # print(data_files)
    def __len__(self):
        return len(self.list_labels)
        

    def __getitem__(self, index):
        image = self.list_images[index]
        image = np.reshape(image,(3,32,32))
        image = np.transpose(image, (1,2,0))
        label = self.list_labels[index]
        return image, label



if __name__ == "__main__":
    training_dataset = MyCifar10(path="my_dataset/cifar-10-batches-py", train=True)
   
    # image, label = dataset[11000]
    # image = np.reshape(image,(3,32,32))
    # image = np.transpose(image, (1,2,0))
    # image = cv2.resize(image, (320,320))
    # cv2.imshow("test", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) #cv2 need BGR to show
    # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()
    # print(image.shape,label)

    # training_dataset = CIFAR10(root="my_dataset", train=True, download=False)
    training_dataloader = DataLoader(
        dataset = training_dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        drop_last=False,
    )
    print(type(training_dataloader))
    for images,labels in training_dataloader:
        print(images.shape,labels.shape)
        