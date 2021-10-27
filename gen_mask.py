from paddleocr import PaddleOCR, draw_ocr
import glob
import os
import cv2
import numpy as np
import shutil

ocr_det_model_dir = './det_r50_vd_inference'
result_img_path = './original_img_and_mask'

# for PaddleOCR constructor, but not used
ocr_cls_model_dir = './ch_ppocr_mobile_v2.0_cls_infer'
ocr_rec_model_dir = './ch_PP-OCRv2_rec_infer'

def ocr_img_mask(original_img_path, window):
    if os.path.exists(result_img_path):
        shutil.rmtree(result_img_path)
    os.makedirs(result_img_path) 

    print("start get img and mask")
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    # ocr = PaddleOCR(det_model_dir=ocr_det_model_dir, use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
    ocr = PaddleOCR(det_model_dir=ocr_det_model_dir,
                    cls_model_dir=ocr_cls_model_dir,
                    rec_model_dir=ocr_rec_model_dir,
                    use_angle_cls=False, lang="ch")
    imgs_path = glob.glob(original_img_path + '/*.jpg')
    num = len(imgs_path)
    i = 0
    for img_path in imgs_path:
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
        i += 1
        window.write_event_value('-PROGRESS-', i / num * 25)

    i = 0
    for img_path in glob.glob(original_img_path + '/*.jpg'):
        img_name = img_path.split('/')[-1]
        # copy original img
        shutil.copy(img_path, os.path.join(result_img_path, img_name))
        i += 1
        window.write_event_value('-PROGRESS-', 25 + i / num * 25)
    
    window.write_event_value('-OCR THREAD-', result_img_path)

# if __name__ == "__main__":
#     ocr_img_mask()
#     print("Finish!")
