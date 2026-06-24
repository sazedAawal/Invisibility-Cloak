# Invisibility-Cloak

A real-time invisibility cloak using OpenCV and MediaPipe that continuously learns the background by filtering out the user's pixels in every frame.

Inspired by the [Ghost / Invisibility Mode](https://github.com/tubakhxn/Invisibility-Computer-Vision) project by **@tubakhxn**.

---

## Controls

While running, click on the video window and use these hotkeys:

* **<kbd>SPACEBAR</kbd>** : Toggle cloak (Switch between `VISIBLE` and `INVISIBLE`)
* **<kbd>R</kbd>** : Wipe and relearn the background (Useful if lighting changes or the camera moves)
* **<kbd>Q</kbd>** : Quit application

---

## Version 1.0
This project uses real-time computer vision and semantic segmentation to create an invisibility cloak. Instead of capturing a static background, 
before the user enters the frame, this initial version continuously checks for user presence and only updates its background model using pixels where 
the user isn't standing. 

N.B.: No stepping out of the frame required—just move around a little at the start so the camera can see what is behind!

---

## Installation

Make sure Python is installed, then the following command to install the required libraries:

```bash
pip install opencv-python mediapipe numpy
