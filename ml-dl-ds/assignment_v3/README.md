## Pipeline :
1. Preprocesing the image (A lot!)
2. RCNN based bounding box predictor, EAST text detector
3. OpenCV Tesseract using LSTM for text-prediction

## Usage :
-  python text_recognition.py --east frozen_east_text_detection.pb --image data-set/ms-7.jpg --padding 0.1

## Output :
- Campus type is printed on screen.

<img src="https://github.com/JaynilJaiswal/sandbox/blob/master/ml-dl-ds/assignment_v3/detectable_text.jpg" width="500">


## Issues :
- Works with only better quality scans.(Like ms-7.jpg & ms-9.jpg)
- Output may depend on the format of how table of marks is arranged. But that code can be changed acoordingly because OCR is working good.
- Only 1 input image at a time.

## Comments :
- I apologize that it is kind of incomplete but given time, I can create full-fledged application.
- Once I join for internship, I can completely deploy it to integrate it with chatbot.
