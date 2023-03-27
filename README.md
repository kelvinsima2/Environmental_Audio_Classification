# Environmental_Audio_Classification
This project classifies 50 different classes of environmental audio sounds using the publicly available [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The classification model consists of neural networks and the pre-trained [InceptionV3 model](https://keras.io/api/applications/inceptionv3/).


# Data Preparation
Data was prepared as detailed in the [code](https://github.com/kelvinsima2/Environmental_Audio_Classification/blob/main/Environmental_Sound_Classification.ipynb). The audio files were converted into mel-spectograms and stored in google drive as images. These images were then trained for classification.

# Model
The deep learning framework used in this project is Tensorflow. The model is summarized as follows (the base model is the [InceptionV3(https://keras.io/api/applications/inceptionv3/)):

* inputs = tf.keras.Input(shape=IMG_SHAPE)
* x = preprocess_input(inputs)
* x = base_model(x, training=False)
* x = tf.keras.layers.BatchNormalization(renorm=True)(x)
* x = tf.keras.layers.GlobalAveragePooling2D()(x)
* x = tf.keras.layers.Dropout(0.5)(x)
* outputs = tf.keras.layers.Dense(50, activation='softmax')(x)
* model = tf.keras.Model(inputs, outputs)

# Results
Overall, the testing accuracy for the model was 59.38%. Most likely, the noise in some of the audio data may have caused the low accuracies. More pre-processing methodologies will be explored to increase the validation and testing accuracies. More ways of reducing overfitting will also be explored. The training and validation accuracy and loss graphs are shown below: <br />
![Accuracy and Loss Graphs](/images/accuracy_audio.png)







