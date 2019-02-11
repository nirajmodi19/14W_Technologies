# 14W_Technologies
Assignment of 14W Technologies to discriminate between images whether the given image is of "VadaPav or Not"


**Method Used** :- Transfer Learning

**Model Used**  :- VGG16

**High Level Wrapper** :- Keras

**Data** :- VadaPav Images(Positive Image), Not VadaPav Images(Negative Images)


**Addressing Directories and Files**:-

1. ./data           :- Contain Images in two sub-directories labelled as "VadaPav" and "notVadaPav".
2. ./test           :- Contain two image for performing the test.
3. ./train.py       :- Python Script to train the model.
4. ./test.py        :- Python Script to test the model.
5. ./VadaPav.model  :- Trained Model

**Steps to use repositories** :-

1. Clone and Unzip the repository.

2. Open Terminal and move to the local directory where you unzipped the repository.

3. Type the command ```python test.py -i test/test1.jpg```


Here in the above command pass the location of the image to be tested(Here ```./test/test1.jpg```)

A display will occur in the console telling you that the image is of **VadaPav or Not with the prediction probability.**
