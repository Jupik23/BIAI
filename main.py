import os
import argparse
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
import importlib
from PIL import Image, ImageTk
from utils.predict import predict_image


class LungClassifier:
    def __init__ (self, root):
        self.root = root
        self.root.title('Binary Classifier')
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ["healthy", "pneumonia"]

    def setup_gui(self):
        main = tk.Frame(self.root, padx=15, pady=15)
        main.pack()
        model = tk.Label(main, text="Choose model")
        model.pack()
        self.model_var = tk.StringVar(value="resnet_pretrained")
        models = {
            "Pretrained ResNet18": "resnet_pretrained",
            "ResNet18 (not pretrained)": "resnet_not_pretrained",
            "ResNet18 + Dropout": "resnet_dropout",
            "ResNet18 + 2 Dropouts": "resnet_2_dropouts_checkpoint",
            "Custom Model": "custom"
        }

        for (text, value) in models.items():
            tk.Radiobutton(main, text=text, variable=self.model_var, value=value).pack(anchor=tk.W)

        load_model_button = tk.Button(main, text="Load Model", command=self.load_model)
        load_model_button.pack(pady=5)

        select_image_button = tk.Button(main, text="Choose image", command=self.select_image)
        select_image_button.pack(pady=5)

        self.image_label = tk.Label(main)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(main, text="Result: ", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

    def load_model(self):
        model_name = self.model_var.get()
        try:
            model_module = importlib.import_module(f"models.{model_name}")
            self.model = model_module.get_model(num_classes=2)
            self.model.to(self.device)
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror(e)

    def select_image(self):
        if not self.model:
            messagebox.showwarning("Choose model!")
        file_path=filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.classify_image(file_path)
    
    def display_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def classify_image(self, file_path):
        try:
            result = predict_image(self.model, file_path, self.device, self.class_names)
            self.result_label.config(text=f"Result: {result}")
        except Exception as e:
            messagebox.showerror("ERROR!", f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungClassifier(root)
    app.setup_gui()
    root.mainloop()