import tkinter as tk
from PIL import ImageTk, Image
import cv2


root = tk.Tk()
# Create a frame
app = tk.Frame(root, bg="white")
app.grid()
# Create box for video
video_box = tk.Label(app)
video_box.grid()
# Create box for exercise instructions
instructions_box = tk.Label(app)
instructions_box.grid()

# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)

    # Add image to box
    imgtk = ImageTk.PhotoImage(image=img)
    video_box.imgtk = imgtk
    video_box.configure(image=imgtk)
    video_box.after(1, video_stream)

def updateInstructions():
    exercise = tk.Label(instructions_box, text="Pushups")
    exercise.pack()
    count = tk.Label(instructions_box, text="10")
    count.pack()
    #instructions_box.after(1, updateInstructions())

video_stream()
updateInstructions()
root.mainloop()