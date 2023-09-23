# Body-Language-Detection-with-MediaPipe-and-OpenCV
**This Jupyter Notebook (IPython Notebook) provides the code and instructions for implementing body language detection using [MediaPipe](https://github.com/google/mediapipe) and [OpenCV](https://github.com/opencv/opencv). This innovative tool incorporates two distinct models to achieve its functionality, providing users with a comprehensive approach to body language analysis.**
### 1. Scikit-Learn (.pkl)
The first model is built using **Scikit-Learn** and is stored in a **.pkl (Python Pickle) format**.
1. It employs pipelines to encapsulate preprocessing and modeling steps for multiple algorithms.
   i. LogisticRegression
   ii. RidgeClassifier
   iii. RandomForestClassifier
   iv. GradientBoostingClassifier
   ```python
   pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
```
3. It systematically trains and evaluates different models using accuracy as a metric.
4. It saves the best-performing model for later use using pickle.
### 2. TensorFlow-Keras (.tflite)

The second model is built using **TensorFlow-Keras** and is stored in a **TensorFlow Lite (.tflite) format**. TensorFlow-Keras is renowned for its deep learning capabilities, and the TensorFlow Lite format ensures that this model can be seamlessly integrated into various applications while maintaining low latency and optimal performance.


