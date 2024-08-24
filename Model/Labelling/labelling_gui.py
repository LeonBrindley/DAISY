import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, Scrollbar, Canvas
import time

class ImageClassifier:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")

        # Set the window size and make it resizable
        self.master.geometry('1000x800')
        self.master.resizable(True, True)

        # Create a canvas and a scrollbar for the segments
        self.canvas = Canvas(self.master)
        self.scrollbar = Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Configure the scrollable region
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        # Add the frame to the canvas and configure the scrollbar
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Load and next buttons
        self.control_frame = tk.Frame(self.master, height=50)
        self.control_frame.pack(fill="x", side="bottom")
        self.load_button = tk.Button(self.control_frame, text="Load Images", command=self.load_images)
        self.load_button.pack(side="left")
        self.next_button = tk.Button(self.control_frame, text="Next Image", command=self.show_image)
        self.next_button.pack(side="left")

        self.directory = ""
        self.image_files = []
        self.current_image = None

    def load_images(self):
        self.directory = filedialog.askdirectory()
        self.image_files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.show_image()

    def show_image(self):
        if self.image_files:
            if self.scrollable_frame.winfo_children():
                for widget in self.scrollable_frame.winfo_children():
                    widget.destroy()
            image_path = self.image_files.pop(0)
            self.current_image = Image.open(image_path)
            self.process_image(self.current_image)

    def process_image(self, img):
        grid_width = img.width // 5
        grid_height = img.height // 4
        self.segments = []

        for i in range(4):
            for j in range(5):
                left = j * grid_width
                top = i * grid_height
                right = left + grid_width
                bottom = top + grid_height
                segment = img.crop((left, top, right, bottom))
                self.segments.append(segment)

        self.display_segments()

    def display_segments(self):
        for i in range(4):
            for j in range(5):
                index = i * 5 + j
                segment = self.segments[index]
                frame = tk.Frame(self.scrollable_frame)
                frame.grid(row=i, column=j, padx=5, pady=5)

                segment = segment.resize((150, 150), Image.ANTIALIAS)  # Resize for better visibility
                tk_img = ImageTk.PhotoImage(segment)

                button1 = tk.Button(frame, image=tk_img)
                button1.image = tk_img
                button1.pack(side="top")

                button2 = tk.Button(frame, text="Clover", command=lambda idx=index: self.save_segment(idx, 'clover'))
                button2.pack(side="left")

                button3 = tk.Button(frame, text="Grass", command=lambda idx=index: self.save_segment(idx, 'grass'))
                button3.pack(side="left")

    def save_segment(self, index, category):
        segment = self.segments[index]
        save_path = os.path.join(self.directory, category)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # add timestamp to the file name
        timestamp = int(time.time())

        file_name = f'segment_{index}_{category}_{timestamp}.png'
        segment.save(os.path.join(save_path, file_name))
        print(f'Segment saved in {save_path} as {file_name}')

def main():
    root = tk.Tk()
    app = ImageClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()
