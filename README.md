# Body-Language-Detection-with-MediaPipe-and-OpenCV
https://github.com/ThisIs-Developer/Body-Language-Detection-with-MediaPipe-and-OpenCV/assets/109382325/231cfaf5-f1e1-410d-998c-caa984f03c52


**This Jupyter Notebook (IPython Notebook) provides the code and instructions for implementing body language detection using [MediaPipe](https://github.com/google/mediapipe) and [OpenCV](https://github.com/opencv/opencv). This innovative tool incorporates two distinct models to achieve its functionality, providing users with a comprehensive approach to body language analysis.**
 * [Scikit-Learn (.pkl)](https://github.com/ThisIs-Developer/Body-Language-Detection-with-MediaPipe-and-OpenCV/blob/main/README.md#1-scikit-learn-pkl)
 * [TensorFlow-Keras (.tflite)](https://github.com/ThisIs-Developer/Body-Language-Detection-with-MediaPipe-and-OpenCV#2-tensorflow-keras-tflite)
 *  [Features ⭐](https://github.com/ThisIs-Developer/Body-Language-Detection-with-MediaPipe-and-OpenCV#features-)

## 1. Scikit-Learn (.pkl)
The first model is built using **Scikit-Learn** and is stored in a **.pkl (Python Pickle) format**.
1. It employs pipelines to encapsulate preprocessing and modeling steps for multiple algorithms.
    ```python
   pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }
    ```

2. It systematically trains and evaluates different models using accuracy as a metric.
   ```output
    lr 0.995260663507109
    rc 0.985781990521327
    rf 0.9881516587677726
    gb 0.9928909952606635
    ```
3. It saves the best-performing model for later use using pickle.
   ```python
    with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
   ```
## 2. TensorFlow-Keras (.tflite)

The second model is built using **TensorFlow-Keras** and is stored in a **TensorFlow Lite (.tflite) format**. 
1. It Builds and compiles a neural network model for classification.
   ```python
       model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
   ```
2. It trains the model with relevant metrics. 
3. It converts and saves the model in TensorFlow Lite format for mobile deployment.
   ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("body_language.tflite", "wb").write(tflite_model)
   ```
## Features ⭐
### 1. Create the training dataset using both a **Webcam** and **recording video data (.mp4)**, extracting relevant frames, and annotating those frames with corresponding labels.
#### View Folder: [Video Decoder](https://github.com/ThisIs-Developer/Body-Language-Detection-with-MediaPipe-and-OpenCV/tree/main/Video%20Decoder)
#### Modify the codef for MP4:
```python
class_name = "Happy"
# Replace 'path_to_your_video_file' with the actual path to your video file
cap = cv2.VideoCapture('path_to_your_video_file') 
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
```
### 2. Trained the machine learning models to recognize 10 different body language and facial expression.
#### Class Labels
1. Happy
2. Sad
3. Angry
4. Surprised
5. Confused
6. Tension
7. Surprised
8. Excited
9. Pain
10. Depressed

