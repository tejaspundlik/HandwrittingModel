import cv2
import typing
import numpy as np
import tf2onnx

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
line_list = []

# img = cv2.imread("C:\\Users\\priya\\Downloads\\workpls.png")
img = cv2.imread(r"C:\Users\priya\Downloads\image1.jpg")
cv2.imshow('image',img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("image",img)
def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh


thresh_img = thresholding(img)
# cv2.imshow("Thresh", thresh_img)
kernel_horizontal = np.ones((1, 5), np.uint8)
lines_image = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel_horizontal, iterations=10)
# cv2.imshow("lines_img",lines_image)
kernel = np.ones((1, 150), np.uint8)
dilated = cv2.dilate(thresh_img, kernel, iterations=1)
cv2.imshow("Dilated", dilated)


(contours, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

for i, ctr in enumerate(contours):
    x, y, w, h = cv2.boundingRect(ctr)
    line_list.insert(0, [x, y, x + w, y + h])
contour_areas = [cv2.contourArea(cnt) for cnt in contours]
min_contour_area = int(np.mean(contour_areas))  # You can also use np.median() if desired

# Filter contours based on area (eliminate relatively small contours)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt)*2.5 > min_contour_area]

contour_image = img.copy()
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h+2), (0, 255, 0), 2)

cv2.imshow('cntr',contour_image) # -1 means draw all contours, (0, 255, 0) is the color, 2 is thickness
img_list = []
for i, line in enumerate(line_list):
    x, y, x_w, y_h = line
    roi = img[y:y_h, x:x_w]
    img_list.append(roi.copy())
    # cv2.imshow(f"", roi)
    # cv2.imwrite()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(len(img_list))



class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])
        # cv2.imshow("c...",image)
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # configs = BaseModelConfigs.load(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Models\configs.yaml")
    #mltu wala model ka configs
    # configs = BaseModelConfigs.load(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202402072234\configs.yaml")
    #mltu wala ka configs
    configs = BaseModelConfigs.load(r"C:\Users\priya\OneDrive\Desktop\Khudka Model\tejas\configs.yaml")
    # from tensorflow.python.keras.models import load_model
    #mltu wala model
    # model = load_model(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Tutorials\04_sentence_recognition\Models\04_sentence_recognition\202402072234\model.h5")
    #hamara model
    # model = load_model(r"C:\Users\priya\OneDrive\Desktop\Khudka Model\HandwritingRecognition\Models\04_sentence_recognition\202403171133\model.h5")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    # df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Models\val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    # for image_path, label in tqdm(df):
        # prediction_text = ""
        # new_image_path = "C:\\Users\\priya\\OneDrive\\Desktop\\PYTHONLESSONS-ML\\mltu\\"
        # image = cv2.imread(new_image_path+image_path)
        # image = cv2.imread("C:\\Users\\priya\\Downloads\\helluuuu.png")

        # print(image_path)
        # for image in img_list:
        #     # cv2.imshow("LiNE",image)
        #     prediction_text += model.predict(image)

        # cer = get_cer(prediction_text, label)
        # wer = get_wer(prediction_text, label)
        # print("Image: ", image_path)
        # print("Label:", label)

        # print(f"CER: {cer}; WER: {wer}")

        # accum_cer.append(cer)
        # accum_wer.append(wer)

        # cv2.imshow(prediction_text, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    prediction_text = ""
    for image in img_list:
        prediction_text += model.predict(image)
    print("Prediction: ", prediction_text)


print("TEJAS :D3 owowowo")
from symspellpy import SymSpell
symsp = SymSpell()
symsp.load_dictionary('corpus.txt',\
                      term_index=0, \
                      count_index=1, \
                      separator=' ')
txt= prediction_text
terms = symsp.lookup_compound(txt,2)
for k in terms:
    print(k.term)