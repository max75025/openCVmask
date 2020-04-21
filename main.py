import cv2
import numpy as np
import os
import glob
#img1 = cv2.imread("./IM000000.png")
#mask1 = cv2.imread("./IM000000_mask.png")


mask_folder = './masks/'


# все изображение исключая маски _mask
#imgs = glob.glob("./*[!_mask].png")
imgs = glob.glob("./*[!_mask].png")
masks = glob.glob(os.path.join(mask_folder, "*.png"))
print("images found:", len(imgs))
print("mask found:", len(masks))


scale_percent = 300 # Процент от изначального размера
alpha = 0.15


#rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # for rgb color



def viewImage(image, name_of_window):
    #cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#viewImage(image, "image")

#img  - you img
# percent from img to scale
def scale(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)








def showAllImg():
    if not os.path.exists(mask_folder):
        print('masks folder not found')
        input()
        os._exit(0)
    if len(imgs)!=len(masks):
        print('different images and masks count')
        input()
        os._exit(0)
    for img_name in imgs:
        img =  cv2.imread( img_name)
        print('open ', img_name)
        #mask = cv2.imread(img_name[:-4] + "_mask.png")
        if not os.path.exists(os.path.join(mask_folder, img_name)):
            print('mask', img_name ,' not found')
            input()
            os._exit(0)

        mask = cv2.imread(os.path.join(mask_folder, img_name))
        #set mask
        dst = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
        resized_img = scale(img, scale_percent)
        resized_mask_img = scale(dst, scale_percent)
        numpy_horizontal_concat = np.concatenate((resized_img, resized_mask_img), axis=1)


        viewImage(numpy_horizontal_concat , img_name)


def showOneImg():
    filename = input('введите имя файла с расширением(.png)\n')
    full_filename = os.path.join('./', filename)
    if len(filename)<6 or not os.path.exists(full_filename):
            print( filename ,' not found')
            input()
            os._exit(0)

    img =  cv2.imread( full_filename)
    print('open ', filename)
    #mask = cv2.imread(img_name[:-4] + "_mask.png")
    if not os.path.exists(os.path.join(mask_folder, filename)):
        print('mask', filename ,' not found')
        input()
        os._exit(0)

    mask = cv2.imread(os.path.join(mask_folder, filename))
    #set mask
    dst = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
    resized_img = scale(img, scale_percent)
    resized_mask_img = scale(dst, scale_percent)
    numpy_horizontal_concat = np.concatenate((resized_img, resized_mask_img), axis=1)

    viewImage(numpy_horizontal_concat , filename)


def showSlideShow(imgs):
    while 1:
        for img_name in imgs:
            img =  cv2.imread( img_name)
            img = scale(img, 300)
            cv2.namedWindow("slideshow", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("slideshow", img) 
            k = cv2.waitKey(67) # 15 fps
            if k==27: #escape
                break    
        if k==27: #escape
            break     
       
       

print('func:')
print('1 - показать все изображения покадрово;')
print('2 - показать одно изображение;')
print('3 - показать слайдшоу(выход esc);')
print('4 - показать слайдшоу масок(выход esc);')

num = int(input("введите цифру:"))

if num == 1:
    showAllImg()
elif num == 2:
    showOneImg()
elif num == 3:
    showSlideShow(imgs)
elif num == 4:
    if not os.path.exists(mask_folder):
        print('masks folder not found')
        input()
        os._exit(0)
    if len(masks) == 0:
        print('masks not found')
        input()
        os._exit(0)
    showSlideShow(masks)
else:
    input('не правильный ввод')
