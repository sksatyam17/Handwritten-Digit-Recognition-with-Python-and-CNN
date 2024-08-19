# Importing modules
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loading model from json file
json_file_path = "trained_model_010921.json"
file = open(json_file_path, 'r')
model_json = file.read()
file.close()
loaded_model = model_from_json(model_json)

# loading weights
h5_file = "weights_010921.hdf"
loaded_model.load_weights(h5_file)

# Testing function
IMG_SIZE = 28
finalResult = None


def testing():
    global finalResult
    a = r"D:\aditi\PycharmProjects\HWDR\Resources\digit.png"
    Img = cv2.imread(a)
    plt.imshow(Img)
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    newImg = tf.keras.utils.normalize(resized, axis=1)
    newImg = np.array(newImg).reshape(-1, 28, 28, 1)
    predictions = loaded_model.predict(newImg)
    finalResult = np.argmax(predictions)
    print(finalResult)


########################################################

# Main Loop

import tkinter
import tkinter.font as font
from PIL import Image, ImageGrab
import pygetwindow
import pyautogui

# Testing function
IMG_SIZE = 28
finalResult = None


def testing():
    global finalResult
    a = r"D:\aditi\PycharmProjects\HWDR\Resources\digit.png"
    Img = cv2.imread(a)
    plt.imshow(Img)
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    newImg = tf.keras.utils.normalize(resized, axis=1)
    newImg = np.array(newImg).reshape(-1, 28, 28, 1)
    predictions = loaded_model.predict(newImg)
    finalResult = np.argmax(predictions)
    print(finalResult)


# Testing Loop
def test():
    save()
    testing()
    result()


# creating window
root = tkinter.Tk()

# window title and size
root.title("HANDWRITTEN DIGIT RECOGNITION")
root.geometry("1150x655")
bg = tkinter.PhotoImage(file=r"D:\aditi\PycharmProjects\HWDR\Resources\Background1.png")
label1 = tkinter.Label(root, image=bg)
label1.place(x=0, y=0)

# building the canvas for drawing image
cv = tkinter.Canvas(root, width="200", height="280", bd='5', bg="black", highlightthickness=0)
cv.place(x=304, y=258, anchor="nw")


def get_x_n_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y


def draw(event):
    global last_x, last_y
    cv.create_line((last_x, last_y, event.x, event.y), fill="white", width=16)
    last_x, last_y = event.x, event.y


cv.bind("<Button-1>", get_x_n_y)
cv.bind("<B1-Motion>", draw)


def save():
    window = pygetwindow.getWindowsWithTitle('HANDWRITTEN DIGIT RECOGNITION')[0]
    x1 = window.left + 323
    y1 = window.top + 313
    height = 280
    width = 200

    x2 = x1 + width
    y2 = y1 + height

    path = r"D:\aditi\PycharmProjects\HWDR\Resources\digit.png"
    pyautogui.screenshot(path)

    im = Image.open(path)
    im = im.crop((x1, y1, x2, y2))
    im.save(path)
    # im.show(path)


def clear():
    cv.delete('all')


def result():
    output_message = tkinter.Button(root, bg='black', fg='white', text=finalResult)
    myFont1 = font.Font(family='Helvetica', size=10, weight='bold')
    output_message['font'] = myFont1
    output_message.place(x=650, y=395, height='40', width='180')


def tryAgain():
    clear()
    global finalResult
    finalResult = None
    result()


def close():
    root.quit()


myFont = font.Font(family='Courier', size=10, weight='bold')
save_button = tkinter.Button(bg='CornflowerBlue', fg='white', text=" Save ", activebackground="LightSkyBlue",
                             command=save)
save_button['font'] = myFont
save_button.place(x=640, y=270)

clear_button = tkinter.Button(bg='CornflowerBlue', fg='white', text=" Clear", activebackground="LightSkyBlue",
                              command=clear)
clear_button['font'] = myFont
clear_button.place(x=745, y=270)

myFont = font.Font(family='Courier', size=12, weight='bold')
test_button = tkinter.Button(bg='DodgerBlue', fg='white', text="  Test  ", activebackground="LightSkyBlue",
                             command=test)
test_button['font'] = myFont
test_button.place(x=670, y=330)

output_message = tkinter.Button(root, bg='black', fg='white', text=finalResult)
myFont1 = font.Font(family='Helvetica', size=10, weight='bold')
output_message['font'] = myFont1
output_message.place(x=650, y=395, height='40', width='180')

tryAgain_button = tkinter.Button(bg='CornflowerBlue', fg='white', text=" Try Again ", activebackground="LightSkyBlue",
                                 command=tryAgain)
tryAgain_button['font'] = myFont
tryAgain_button.place(x=650, y=450)

Quit_button = tkinter.Button(bg='CornflowerBlue', fg='white', text=" Quit ", activebackground="LightSkyBlue",
                             command=close)
Quit_button['font'] = myFont
Quit_button.place(x=680, y=505)

root.mainloop()
