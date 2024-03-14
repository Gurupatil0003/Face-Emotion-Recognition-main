# <h1>Face-Emotion-Recognition.</h1><br>
<p align="center">
<img src="https://github.com/anishjohnson/Face-Emotion-Recognition/blob/main/Images%20Used/github_cover_croped.jpg">
</p>

This project is a part of "Deep Learning + ML Engineering” curriculum as capstone projects at [Almabetter School.](https://www.almabetter.com/)<br>

## The required files can be accessed through the below links directly:
* [Colab Notebook](https://github.com/anishjohnson/Face-Emotion-Recognition/blob/f2e1968ebdf82ddccb5e42168d1fb48de44f9b1c/FER/Colab%20Notebook/Face_Emotion_Recognition_Anish_Johnson.ipynb)
* [Streamlit App (.py) File](https://github.com/anishjohnson/Face-Emotion-Recognition/blob/f2e1968ebdf82ddccb5e42168d1fb48de44f9b1c/streamlit_app.py)
* [Project Presentation](https://github.com/anishjohnson/Face-Emotion-Recognition/blob/f2e1968ebdf82ddccb5e42168d1fb48de44f9b1c/Project%20Presentation/Face%20Emotion%20Recognition%20Presentation.pdf)



## Project Introduction:<br>
<p>The Indian education landscape has been undergoing rapid changes for the past ten years owing to the advancement of web-based learning services, specifically eLearning platforms.</p>

<p>Digital platforms might overpower physical classrooms in terms of content quality, but in a physical classroom, a teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.
While digital platforms have limitations in terms of physical surveillance, it comes with the power of data and machines, which can work for you.</p>

<p>It provides data in form of video, audio, and texts, which can be analyzed using deep learning algorithms. A deep learning-backed system not only solves the surveillance issue, but also removes the human bias from the system, and all information is no longer in the teacher’s brain but rather translated into numbers that can be analyzed and tracked.</p>

## Objective:<br>
Our objective is to solve the above mentioned challenge by applying deep learning algorithms to live video data inorder to recognize the facial emotions and categorize them accordingly.

## Dataset used:<br>
We have utilized the [FER 2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset provided on Kaggle.<br>
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.<br>

## Dependencies:<br>
<p> Install these libraries before running the colab notebook.</p>
1. numpy<br>
2. streamlit==1.9.0<br>
3. tensorflow-cpu==2.9.0<br>
4. opencv-python-headless==4.5.5.64<br>
5. streamlit-webrtc==0.37.0<br>

## Project Overview:<br>
<p>We start with downloading the required dataset from Kaggle. Once the data is available, the training and validation sets are defined.</p>

<p>The next step is to preprocess the datasets; this includes rescaling the data by multiplying it by 1/255 to obtain the target values in the range [0,1] and performing data augmentation for artificially creating new data. Data augmentation also helps to increase the size and introduce variability in the dataset.</p>

<p>After preparing the data, we construct the required CNN model using TensorFlow and Keras libraries to recognize the facial emotions of a user. This model consists of four convolutional layers and three fully connected layers to process the input image data and predict the required output. In between each layer, a Max Pooling and Dropout layer was added for downsampling the data and preventing our model from overfitting. Finally, for compiling all the layers, we have used the Adam optimizer, with loss function as Categorical Cross entropy and accuracy as the metric for evaluation.</p>

<p>Once the model was ready, we trained it using the prepared data.</p>

<p>The model achieved an accuracy of 77% on the training set and 64% on the validation set after fifteen epochs.</p>
<img src='https://github.com/anishjohnson/Face-Emotion-Recognition/blob/main/Images%20Used/loss%20%26%20accuracy.png'>

<p>From the confusion matrix, we saw; that the model accurately predicts most classes, but the performance is comparatively lower in classes angry and fear. Less amount of data present for these classes might be the reason for this.</p>
<img src='https://github.com/anishjohnson/Face-Emotion-Recognition/blob/main/Images%20Used/confusion%20matrix.png'>

<p>Finally, using an image passed through our model, we confirmed that it could correctly recognize the emotions.</p>

<p>Additionally, using OpenCV and Streamlit, we created a web app to monitor live facial emotion recognition. The web app successfully identified each class and also was able to detect multiple faces and their respective emotions.</p>

The link for the web app: https://share.streamlit.io/anishjohnson/face-emotion-recognition/main

## Live Facial Emotion Recognition:
<img src='https://github.com/anishjohnson/Face-Emotion-Recognition/blob/main/Images%20Used/face_detect.png'>

## Demo video to display the working of our web app deployed on streamlit.

https://user-images.githubusercontent.com/87002524/170706372-9cadcd73-2f1e-41d2-b61a-d9b8042cbd95.mp4


## For reference:<br>
* https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de
* https://towardsdatascience.com/video-facial-expression-detection-with-deep-learning-applying-fast-ai-d9dcfd5bcf10
* https://github.com/atulapra/Emotion-detection
* https://medium.com/analytics-vidhya/building-a-real-time-emotion-detector-towards-machine-with-e-q-c20b17f89220
