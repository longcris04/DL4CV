import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import shutil
from sklearn.model_selection import train_test_split
import cv2
import json
"""
    Không dùng ImageFolder nhé
"""
class AnimalDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            data_path = os.path.join(root,"train")
        else:
            data_path = os.path.join(root, "test")
        self.imgs = []
        self.labels = []
        for Class in os.listdir(data_path):
            Class_path = os.path.join(data_path,Class)
            for img_name in os.listdir(Class_path):
                img_path = os.path.join(Class_path,img_name)
                # print(img.shape)
                self.imgs.append(img_path)
                self.labels.append(Class)


        
    def __len__(self):
        
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        print(image_path)
        image = cv2.imread(image_path)

        label = self.labels[idx]
        return image, label


"""
    Giới thiệu bộ dataset football
    - Mỗi 1 video kéo dài khoảng 1 phút, với 25 khung hình 1 giây (25 FPS)
    - Mỗi 1 video sẽ có 1 file annotation .json tương ứng (Mình khuyến khích các bạn mở các file này ra và
    thử cố gắng hiểu về các attribute trong các file này nhé)
    - Đối với từ khóa "categories": Nhìn chung sẽ có 4 đối tượng được annotate trong các video, với ID là 1 cho đến 4. Tạm thời các bạn
    chỉ cần quan tâm đến id = 4, là các cầu thủ là được
    - Đối với từ khóa "images": Đây là thông tin về các frame trong video. Các bạn chú ý là ở đây frame xuất phát từ 1,
    nhưng trong lập trình chỉ số xuất phát từ 0 nhé. Nhìn chung sẽ có 1500 frames, tương ứng với 1 phút
    - Đối với từ khóa "annotations": ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT. Các bạn sẽ thấy trong trường này có rất nhiều
    dictionary, mỗi 1 dictionary tương ứng với 1 object trong 1 frame nhất định, trong đó:
        + id: không cần quan tâm
        + image_id: id của frame (chạy từ 1 cho đến 1500)
        + category_id: Các bạn chỉ cần quan tâm đến những item mà category_id = 4 (player) là được
        
    TASK: Các bạn hãy xây dựng Dataset cho bộ dataset này, với các quy tắc sau:
    - Hàm __init__ tùy ý các bạn thiết kế
    - Hàm __len__ trả về tổng số lượng frame có trong tất cả các video
    - Hàm __getitem__(self, idx) trả về list của các bức ảnh đã được crop về các cầu thủ (trong hầu hết các frame
    là sẽ có 10 cầu thủ trong 1 frame) và list các số áo tương ứng của các cầu thủ này. idx sẽ theo quy tắc sau: Giả sử
    các bạn gộp tất cả các video thành 1 video dài (thứ tự các video con tùy các bạn), thì idx sẽ là index của video 
    dài đó. Ví dụ trong trường hợp chúng ta có 3 video con dài 1 phút, thì video tổng sẽ dài khoảng 3 phút và có
    4500 frames tổng cộng.
    
    GOOD LUCK!
"""



class FootballDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.jsons_path = []
        self.videos_path = []
        self.length = 0
        self.chunks = [] # store length (number of frames) for each videos in subfolder
        
        if train:
            path = os.path.join(root,"football_train") 
        else:
            path = os.path.join(root,"football_test")

        subfolders = os.listdir(path)
        for subfolder in subfolders:
            path_subfolder = os.path.join(path,subfolder)
            jsons_paths = [os.path.join(path_subfolder,file) for file in os.listdir(path_subfolder) if file.endswith('.json') ]
            videos_paths = [os.path.join(path_subfolder,file) for file in os.listdir(path_subfolder) if file.endswith(('.mp4', '.mkv', '.avi', '.mov'))]
            if len(jsons_paths) == len(videos_paths):
                self.jsons_path.extend(jsons_paths)
                self.videos_path.extend(videos_paths)
        
        for json_path in self.jsons_path:
            with open(json_path, "r") as file:
                data = json.load(file)
                chunk = len(data['images'])
                self.chunks.append(chunk)
                self.length += chunk
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        player_images = []
        jersey_numbers = []
        
        if idx < 0 or idx >= self.length: # Raise error if input index out of range
            raise IndexError(f"Index out of bounds: {idx}. Valid range is 0 to {self.length - 1}.")
        else:
            # find which chunk (0/1/2/3) the global index (0-6040) is in and update it to be local idx (starting from 0) 
            chunk = -1
            accumulate = 0
            for i in range(len(self.chunks)):
                cur = accumulate
                accumulate += self.chunks[i]
                if (cur <= idx < accumulate):
                    chunk = i
                    idx = idx - cur + 1 # update idx = local index 
                    break

        # Read video by using chunk as index
        video_path = self.videos_path[chunk]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            exit()
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames in the video: {total_frames}")  # This code for check total frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)               # find the frame by index
        ret, frame = cap.read() 
        if not ret:
            raise ValueError("Cannot read frame from the video.")
            exit()
        cap.release()


        # find boundary of players and associated jersey number
        json_path = self.jsons_path[chunk] # find path to the json file
        with open(json_path, "r") as file:
            data = json.load(file)
        img_start_idx = self.binary_search(data['annotations'],idx)     # Find the starting index of first object appeared in the frame number

        # Find all objects with category_id == 4 within the frame's index range
        i = img_start_idx
        while True:
            if i >= len(data['annotations']) or data['annotations'][i]['image_id'] > idx: # Break the while loop if the index is outside that frame's index range
                break
            if data['annotations'][i]['category_id'] == 4:
                x,y,w,h = data['annotations'][i]['bbox']  
                crop = frame[int(y):int(y+h),int(x):int(x+w)]
                player_images.append(crop)

                number = data['annotations'][i]['attributes']['jersey_number']
                jersey_numbers.append(number)
                cv2.imshow("crop img", crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            i += 1
        
        



        return player_images, jersey_numbers

    def binary_search(self,data,value):    # Find left most index
        left, right = 0, len(data) - 1
        res = -1
        while left <= right:
            mid = (left + right) // 2
            if data[mid]['image_id'] < value:
                left = mid + 1
            elif data[mid]['image_id'] > value:
                right = mid - 1
            else: 
                res = mid
                right = mid - 1
        return res

class CIFARDataset(Dataset):
    def __init__(self, root=".", train=True, transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        label = self.labels[item]
        return image, label


if __name__ == '__main__':

    # football dataset

    path = os.path.join(os.getcwd(),"../Datasets")
    dataset = FootballDataset(root = path, train=False, transform=None)
    # print("length:",len(dataset))
    player_images, jersey_numbers = dataset[2000]
    print(jersey_numbers)
    for i in player_images:
        print(i)

    

    
    # a = [os.path.isdir(os.path.join(animal_path,i)) for i in a]
    # print(a)

    # dataset = CIFARDataset(root=".")
    # index = 400
    # image, label = dataset.__getitem__(index)
    # print(image.shape)
    # print(label)

    # Split animal dataset into train test
    
    # train_ratio = 0.8
    # animal_path = os.path.join(os.getcwd(),"../Datasets/animals")
    # animal_split_path = os.path.join(os.getcwd(),"../Datasets/animals_split")

    # train_folder = os.path.join(animal_split_path,"train")
    # test_folder = os.path.join(animal_split_path,"test")
    # os.makedirs(train_folder,exist_ok=True)
    # os.makedirs(test_folder,exist_ok=True)

    # for Class in os.listdir(animal_path):
    #     class_path = os.path.join(animal_path,Class)
    #     if os.path.isdir(os.path.join(animal_path,Class)):
    #         # mkdir in train and test folder
    #         Class_train_folder = os.path.join(train_folder,Class)
    #         Class_test_folder = os.path.join(test_folder,Class)
    #         os.makedirs(Class_train_folder,exist_ok=True)
    #         os.makedirs(Class_test_folder,exist_ok=True)

    #         # find all image in original folder
    #         images = [os.path.join(class_path,image) for image in os.listdir(class_path) if os.path.isfile(os.path.join(class_path,image))] 
            
    #         train_images, test_images = train_test_split(images,train_size=train_ratio,random_state=42)
            
    #         for img in train_images:
    #             shutil.copy(img,Class_train_folder)
                
    #         for img in test_images:
    #             shutil.copy(img,Class_test_folder)
               
    #         # print(len(os.listdir(os.path.join(Class_train_folder))))
    #         # print(len(os.listdir(os.path.join(Class_test_folder))))

    # dataset = AnimalDataset(root=os.path.join(os.getcwd(),"../Datasets/animals_split"),train=False)
    # index = 4005
    # image, label = dataset[index]
    # print(image.shape,label)
    # print(len(dataset))
    # cv2.imshow("sample",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
