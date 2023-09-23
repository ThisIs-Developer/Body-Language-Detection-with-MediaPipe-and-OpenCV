# Body-Language-Detection-with-MediaPipe-and-OpenCV
**This Jupyter Notebook (IPython Notebook) provides the code and instructions for implementing body language detection using [MediaPipe](https://github.com/google/mediapipe) and [OpenCV](https://github.com/opencv/opencv). This innovative tool incorporates two distinct models to achieve its functionality, providing users with a comprehensive approach to body language analysis.**
### 1. Scikit-Learn (.pkl)
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
### 2. TensorFlow-Keras (.tflite)

The second model is built using **TensorFlow-Keras** and is stored in a **TensorFlow Lite (.tflite) format**. TensorFlow-Keras is renowned for its deep learning capabilities, and the TensorFlow Lite format ensures that this model can be seamlessly integrated into various applications while maintaining low latency and optimal performance.


