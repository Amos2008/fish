import random

with open('testok.txt', 'r') as file:
    image_addresses = file.readlines()


random.shuffle(image_addresses)

with open('test.txt','w',encoding='UTF-8') as f:
    for train_img in image_addresses:
        f.write(str(train_img))