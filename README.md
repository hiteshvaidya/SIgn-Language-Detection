# Sign-Language-Detection
This program detects various hand gestures captured through video input

![live output](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/images/live%20output.png)

### Introduction
Sign language detection is an interesting problem with real-time application. In this repository, we see how we can detect a hand gesture using deep learning. Following are some of the highlights of this project.
- Create a small dataset of images captured through a webcam
- Annotate these images
- Use transfer learning for object detection to train a model on these images
- Detect hand gestures in real time

### Design decisions and structure
- This project uses one of the many available pre-trained convolutional neural networks from Tensorflow object detection model zoo
- The selection of a model from this model zoo could be made based on a tradeoff between speed and performance
- For this particular implementation, I have selected 'SSD MobileNet V2 FPNLite 320x320' from the available list
- The aim was to run a quick dirty model which gives some results. With the availability of resources, this project could be scaled up to various datasets and higher accuracy could be obtained with more sophisticated models

### Pre-requisites
If you are running this on a local machine, I recommend setting up following environment
- Make a virtual environment using either pip or conda
- Install jupyter notebook, tensorflow, opencv-python
- At any point, if you face any import errors, please install the respective packages according to the requirement

If you are running these notebooks on google colab, you need not install anything specifically

### Steps to reproduce
**Step 1**: Git clone this repository [https://github.com/hiteshvaidya/Sign-Language-Detection.git](https://github.com/hiteshvaidya/Sign-Language-Detection.git)

**Step 2**: Setup a virtual environment using either venv or conda and install jupyter notebook in it. Also, create a kernel for jupyter notebook belonging to your virtual environment. This could be done as follows,
```
pip install ipykernel
python -m ipykernel install --user --name=tfodj
```

**Step 3**: Build a small dataset by capturing images via webcam. Use Notebook [1. Image Collection.ipynb](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/1.%20Image%20Collection.ipynb) for this step. Make sure you change the kernel setting in jupyter notebook to your virtual environment
![kernel](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/images/change%20kernel.png)

**Step 4**: Divide the captured images and their corresponding bounding box metadata into `train` and `test` folders. For large number of images you may write a script that performs this job
`Object Detection\Tensorflow\workspace\images\train`
`Object Detection\Tensorflow\workspace\images\test`

**Step 5**: Train the model using notebook, [2. Training and Detection.ipynb](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/2.%20Training%20and%20Detection.ipynb).

**Step 6**: During this process, you will install tensorflow object detection API. Ensure that this step outputs 'OK'. If it throws any errors, install those packages manually and again verify whether all the packages are installed using this step
![verify](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/images/verify.PNG)

**Step 7**: Further, train the model and you may evaluate it and use tensorboard for visualization

### Test output
![test output](https://github.com/hiteshvaidya/Sign-Language-Detection/blob/main/images/test%20output.png)

### Testing it live
- If you are running this code on local machine, you can run the cell in notebook as it is.
- If you are running this on google colab, it is a headache to get live video from webcam. Therefore, I just captured an image on google colab and gave it as in input to the model and got the prediction. Following are the changes that I made to the cell for live detection,
```
# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap = frame
height, width, channels = input_image.shape

# while cap.isOpened(): 
# ret, frame = cap.read()
image_np = np.array(input_image)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

# cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
cv2_imshow(image_np_with_detections)
# if cv2.waitKey(10) & 0xFF == ord('q'):
#     cap.release()
#     cv2.destroyAllWindows()
#     break
```
The output is as shown in the topmost picture of this README,

### Shipping to production
You may follow the code given in other [directory](https://github.com/hiteshvaidya/Sign-Language-Detection/tree/main/App) for shipping this code to production

### Credits
![Nicholas Renotte](https://github.com/nicknochnack)
