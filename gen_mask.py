from paddleocr import PaddleOCR, draw_ocr
import glob
import os
import cv2
import numpy as np
import shutil

original_img_path = '/path/to/your/images/folder'
ocr_det_model_dir = './det_r50_vd_inference'

result_img_path = '/path/to/RemoveText/lama/original_and_mask'

if os.path.exists(result_img_path):
    shutil.rmtree(result_img_path)
os.makedirs(result_img_path)  

def ocr_img_mask():
    print("start get img and mask")
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(det_model_dir=ocr_det_model_dir, use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory

    for img_path in glob.glob(original_img_path + '/*.jpg'):
        img_name = img_path.split('/')[-1]
        result = ocr.ocr(img_path, cls=False)
        # for line in result:
        #     print(line)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        # read img
        image = cv2.imread(img_path)
        h, w, c = image.shape
        mask = np.zeros((h, w), np.uint8)
        #mask = mask * 255

        # draw mask
        if len(boxes):
            for box in boxes:
                lt = tuple(map(int, box[0]))
                br = tuple(map(int, box[2]))

                # extend box(my textbox is too small due to labeling issue)
                ymin = int(max(lt[1] - (br[1] - lt[1]) / 3, 0))
                ymax = int(min(br[1] + (br[1] - lt[1]) / 3, h))
                xmin = int(max(lt[0] - (br[1] - lt[1]) / 3, 0))
                xmax = int(min(br[0] + (br[1] - lt[1]) / 3, w))

                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
        else:
            pass
        
        # write mask
        mask_name = os.path.join(result_img_path, img_name.split('.')[0]+'_mask.jpg')
        cv2.imwrite(mask_name, mask)

    for img_path in glob.glob(original_img_path + '/*.jpg'):
        img_name = img_path.split('/')[-1]
        # copy original img
        shutil.copy(img_path, os.path.join(result_img_path, img_name))

    return

if __name__ == "__main__":
    ocr_img_mask()
    print("Finish!")
