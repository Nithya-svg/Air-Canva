# Air-Canva 🎨✋

Air-Canva is a **computer vision–based mini project** that allows users to draw on a virtual canvas using hand or object movements in front of a webcam. Instead of using a mouse or touch screen, drawing is done **in the air**, making the interaction contact-free and intuitive.

This project is developed using **Python and OpenCV** and demonstrates the practical application of image processing concepts in real time.

---

## 🎯 Objective

The main objective of this project is to:

* Understand **real-time video processing**
* Apply **color detection and contour tracking**
* Create an **interactive drawing application** using computer vision

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **NumPy**

---

## 📌 Features

* Real-time webcam input
* Draw on screen using a colored object or finger
* Virtual canvas for drawing
* Simple and beginner-friendly implementation
* No physical contact required

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nithya-svg/Air-Canva.git
cd Air-Canva
```

### Step 2: Install Required Libraries

```bash
pip install opencv-python numpy
```

### Step 3: Run the Program

```bash
python air_canvas.py
```

*(If the file name is different, run the respective `.py` file.)*

---

## ⚙️ Working Principle

1. The webcam captures live video frames.
2. Each frame is processed to detect a specific color.
3. The detected object’s movement is tracked using contours.
4. The movement coordinates are mapped onto a virtual canvas.
5. Continuous movement results in drawing on the screen.

---

## 🧪 Requirements

* Python 3.x
* Webcam
* Basic lighting conditions
* A colored object (marker / cap)

---

## 📁 Project Structure

```
Air-Canva/
├── air_canvas.py     # Main Python file
├── canva2.py         # Alternate version
├── README.md         # Project documentation
```

---

## 🎓 Academic Relevance

This project is suitable for:

* Computer Vision mini project
* Python mini project
* OpenCV lab submission
* College project / PEP / Internal assessment

---

## 📜 Conclusion

Air-Canva is a simple yet effective project that showcases how **computer vision** can be used to build interactive applications. It is ideal for college students to gain hands-on experience with OpenCV and real-time image processing.
