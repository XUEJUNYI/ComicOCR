from PIL import Image
import os
from tqdm import tqdm
from yolo import YOLO

if __name__ == "__main__":

    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    dir_intput_path= "input/"
    yolo = YOLO()
    
    
    for root, dirs, files in os.walk(dir_origin_path):
        for img_name in tqdm(files):

            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                relative_path = os.path.relpath(root, dir_origin_path)
                img_name_no_ext = os.path.splitext(img_name)[0]
                save_folder = os.path.join(dir_save_path, relative_path,img_name_no_ext)
                out_folder = os.path.join(dir_intput_path ,relative_path,img_name_no_ext)
 			
                image_path = os.path.join(root, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image,out_folder)
                
               
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    
                save_path = os.path.join(save_folder, img_name.replace(".jpg", ".png"))
                r_image.save(save_path, quality=95, subsampling=0)
