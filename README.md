# KINESIC
This repository contains the work done for the project kinesic under Institute Technical Summer Projects(ITSP) 2020
![alt](https://api.insti.app/static/upload/a6/76/a676957d-956a-4810-b0a3-ae1975f20761-45A56223-8316-4F87-BA26-6D34390C83B6.jpeg.jpg)<br/>
<br/>
The goal was to basically build a system that could replace the conventional input methods like the mouse and keyboard with more natural methods of input like hand gestures and speech.<br/>
<br/>
A webcam attached to the computer takes video and feeds it to the computer program.The program uses basic image processing algorithms like background subtraction,gaussian downscale, morphological transformations,skin mask,camshift,thresholding and histogram equalization to detect the position of the hand in the field of view of the webcam.The part of the image consisting of the face is detected and removed using a cascade classifier to prevent skin mask from mistaking it for hand. We use this image of the segmented hand and find out its centre.This is passed on to a classifier which light weight CNN,It classifies the images into different gestures/null(no gesture),Each gesture is associated with a function Eg- fist is associated with a left click.If the image gets successfully classified the action is performed and the next frame is fed into the system.Tracking the centre helps in moving the cursor,the change in the position of the hand in the field of view is associated with the change in position of the cursor on the screen.The classifier has  a high validation and test set accuracy and the system performs well when the angle between the platform and the screen of the laptop is anywhere between 84 to 100 degrees.<br/>
<br/>
The speech to text conversion program run parallely alongwith the above gesture detection and classification program ,a Pyaudio library dependent program is used to record the speech of the user and then it is converted into text using speech recognition library.The model features an energy threshold dependent noise subtraction at the start of each loop The system provides the user commands to output start and stop the program and also a few special commands to stop the speech to text conversion while still running the gesture detection program.All of this is done in real time.<br/>
<br/>
Dependencies<br/>
Python 3.5.0+<br/>
OpenCV 4.1.0+<br/>
Tensorflow 2.0<br/>
SpeechRecognition 3.8<br/>
PyAutoGUI<br/>
PyAudio<br/>
<br/>
References<br/>
- [Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks--Okan Köpüklü, Ahmet Gunduz, Neslihan Kose, Gerhard Rigoll-arXiv:1901.10323v3](https://arxiv.org/abs/1901.10323)
