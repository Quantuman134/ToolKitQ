import cv2

def dilate(img, se_shape:int=cv2.MORPH_RECT, dist:int=1, iter:int=1):
    se = cv2.getStructuringElement(se_shape, (dist * 2 + 1, dist * 2 + 1), (dist, dist))
    return cv2.dilate(img, se, iterations=iter)

def erode(img, se_shape:int=cv2.MORPH_RECT, dist:int=1, iter:int=1):
    se = cv2.getStructuringElement(se_shape, (dist * 2 + 1, dist * 2 + 1), (dist, dist))
    return cv2.erode(img, se, iterations=iter)

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    img = Image.open('./ToolKitQ/ImgTools/sample/mask.png')
    img_array = np.asarray(img)
    img_dilate = dilate(img_array, se_shape=cv2.MORPH_ELLIPSE, dist=3, iter=3)
    img_erode = erode(img_array, se_shape=cv2.MORPH_ELLIPSE, dist=3, iter=3)
    Image.fromarray(img_dilate).save('./ToolKitQ/ImgTools/sample/mask_dilated.png')
    Image.fromarray(img_erode).save('./ToolKitQ/ImgTools/sample/mask_eroded.png')
    