# video_classification_using_Deep_Learning_Models

### Problem Statment
Human action recognition in video is of interest for applications such as automated surveillance, elderly behavior monitoring, human-computer interaction, content-based video retrieval, and video summarization. For example, in monitoring the activities like drowsiness or fatigue of a driver or not using proper safety measures in Industries.

The main problem with images classification is that we will not be able to detect the continuous action from any images which can be done using video. For example, if someone removes the harness while working in a hazardous situation or a driver yawn continuously 3-4 times while driving. This continuous action will not be able to detect by image classification which is an important action to alarm the person. Video classification can be used for other various purposes like automated surveillance, elderly behavior monitoring or video summarization, and many more.

### Objective

The objective of this study was to build models to perform video classification - one of the most exciting applications of deep learning. 

The dataset considered for this project is the HMDB51 video dataset. A significant portion of the time was spent studying how a video dataset can be loaded, processed, and prepared for its input to the model. In this implementation, the video is split into 16 frames. These frames are stacked; this creates a sample whose shape resembles that of a 5D dataset (i.e. [batch_size, num_of_frames, color_channels, height, width]). In total there are a total of 51 classes. To manage the computation time and resources 4 classes were selected. In the final work, we plan to include a set of pre-defined classes. However, to demonstrate the functioning of the model, 4 classes were selected. The intricate details of the data preparation process and the factors affecting the model performance are highlighted in the later sections.

There are several novel approaches for video classification. The models implemented in this work were inspired by the lab assignments conducted in this course. In addition, certain high-performance models were also explored. The below list summarizes the models implemented.


- **Resnet-18 + LSTM Model:** models.resnet18(pretrained=pretrained) with a combination of LSTM architecture. Both the architectures are a part of the forward function and thus the model weights for both of these networks will be updated during the training process. 

- **Transfer Learning + LSTM + Self-Attention Model:** The first model was modified and learnings from LAB2 were combined to create a new architecture. It was observed that the first model was extremely time-consuming. To exploit the performance of the Resnet18 model, it was not eliminated. Transfer learning was employed. The dataset was first passed onto the pre-trained ResNet model and features were extracted. These features were then saved locally. This too is computationally expensive but should be implemented only once. In this notebook, this approach is demonstrated for four classes. In addition, self-attention mechanism is added alongside the LSTM model to improve the classification performance.

- **Resnet-18 + 3D-CNN Model:** models.video.r3d_18(pretrained=True, progress=False). This model is based on the work presented in this paper (https://arxiv.org/pdf/1711.11248.pdf). Although this concept is not covered in the lectures, we decided to explore its implementation based on its superior performance in the video classification domain. 


**In the duration of this project, some of the steps that were implemented are:**

1. Clean and process the video dataset and create appropriate samples for training the model.
2. Implement and modify high-performance video classification models. In addition, implement hybrid models which exploit the advantages of both the CNN and the RNN architectures.
3. Identify and tune critical hyperparameters to improve model performance (only a few of these iterations are presented in this notebook to limit the computation time and improve the readability of the notebook).
4. Apply transfer learning to reduce time complexity which allows for multiple iterations.
5. Comparative Study of the models on a new test set.
