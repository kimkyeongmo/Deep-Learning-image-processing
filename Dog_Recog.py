import cv2 as cv
import numpy as np
import tensorflow as tf
import os
import pickle
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

cnn = tf.keras.models.load_model('cnn_for_stanford_dogs.h5')
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))

class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('견종 인식')
        self.setGeometry(200, 200, 700, 400)
        self.img = None

        fileButton = QPushButton('강아지 사진 열기', self)
        recognitionButton = QPushButton('품종 인식', self)
        quitButton = QPushButton('나가기', self)

        fileButton.setGeometry(10, 10, 100, 30)
        recognitionButton.setGeometry(120, 10, 100, 30)
        quitButton.setGeometry(230, 10, 100, 30)

        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.label = QLabel(self)
        self.label.setGeometry(10, 50, 500, 300)
        self.label.setAlignment(Qt.AlignCenter)

    def pictureOpenFunction(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '이미지 선택', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            self.img = cv.imread(file_name)
            pixmap = QPixmap(file_name).scaled(500, 300, Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

    def recognitionFunction(self):
        if self.img is None:
            return
        x = np.reshape(cv.resize(self.img, (224, 224)), (1, 224, 224, 3))
        res = cnn.predict(x)[0]
        top5 = np.argsort(-res)[:5]
        top5_dog_species_names = [dog_species[i] for i in top5]

        for i in range(5):
            prob = f"({res[top5[i]]:.2f})"
            name = str(top5_dog_species_names[i]).split('-')[1]
            cv.putText(self.img, prob + " " + name, (10, 100 + i * 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('Dog image', self.img)
        os.system('afplay /System/Library/Sounds/Glass.aiff')

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()
