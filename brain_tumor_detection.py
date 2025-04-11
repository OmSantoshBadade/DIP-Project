# Brain Tumor Detection using Image Processing in Python with GUI

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tkinter import Tk, filedialog, Label, Button, Canvas, PhotoImage, messagebox
from PIL import Image, ImageTk

# -------------------- Image Preprocessing Functions --------------------
def preprocess_image(img_path):
    image = cv2.imread(img_path, 0)  # Load image in grayscale
    if image is None:
        raise ValueError(f"Could not load image at {img_path}")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return image, closing

# -------------------- Feature Extraction --------------------
def extract_features(segmented_img):
    contours, _ = cv2.findContours(segmented_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h
        return [area, perimeter, aspect_ratio]
    else:
        return [0, 0, 0]

# -------------------- Load Dataset --------------------
def load_dataset(folder_path):
    X = []
    y = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dataset directory '{folder_path}' not found. Please create a 'dataset' folder with 'tumor' and 'no_tumor' subfolders.")
    
    for label in ['no_tumor', 'tumor']:
        class_folder = os.path.join(folder_path, label)
        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Class folder '{class_folder}' not found. Please create 'tumor' and 'no_tumor' subfolders in the dataset directory.")
        
        files = os.listdir(class_folder)
        if not files:
            raise ValueError(f"No images found in '{class_folder}'. Please add some images to this folder.")
            
        for file in files:
            file_path = os.path.join(class_folder, file)
            try:
                _, segmented = preprocess_image(file_path)
                features = extract_features(segmented)
                X.append(features)
                y.append(1 if label == 'tumor' else 0)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    if not X:
        raise ValueError("No valid images could be processed from the dataset.")
        
    return np.array(X), np.array(y)

# -------------------- Train and Evaluate --------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    return model

# -------------------- Predict on New Image --------------------
def predict_image(model, img_path):
    original, segmented = preprocess_image(img_path)
    features = extract_features(segmented)
    pred = model.predict([features])[0]
    result = 'Tumor Detected' if pred == 1 else 'No Tumor Detected'
    return original, segmented, result

# -------------------- GUI Code --------------------
def launch_gui(model):
    def upload_and_predict():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        try:
            original, segmented, result = predict_image(model, file_path)

            # Convert OpenCV images to PIL for displaying in Tkinter
            original_img = Image.fromarray(original)
            segmented_img = Image.fromarray(segmented)
            original_img = original_img.resize((200, 200))
            segmented_img = segmented_img.resize((200, 200))

            original_photo = ImageTk.PhotoImage(original_img)
            segmented_photo = ImageTk.PhotoImage(segmented_img)

            original_label.config(image=original_photo)
            original_label.image = original_photo
            segmented_label.config(image=segmented_photo)
            segmented_label.image = segmented_photo
            result_label.config(text=f"Result: {result}", font=("Arial", 16))
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    # Initialize GUI
    root = Tk()
    root.title("Brain Tumor Detection")
    root.geometry("500x500")

    upload_btn = Button(root, text="Upload MRI Image", command=upload_and_predict, font=("Arial", 14))
    upload_btn.pack(pady=20)

    original_label = Label(root)
    original_label.pack()

    segmented_label = Label(root)
    segmented_label.pack()

    result_label = Label(root, text="", font=("Arial", 14))
    result_label.pack(pady=10)

    root.mainloop()

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    try:
        dataset_path = "dataset"  # Folder with 'tumor' and 'no_tumor' subfolders
        print("Loading dataset...")
        X, y = load_dataset(dataset_path)
        print("Training model...")
        model = train_model(X, y)
        print("Launching GUI...")
        launch_gui(model)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTo use this application, you need to:")
        print("1. Create a 'dataset' folder in the same directory as this script")
        print("2. Inside the 'dataset' folder, create two subfolders: 'tumor' and 'no_tumor'")
        print("3. Add brain MRI images to these folders (tumor images in 'tumor' folder, non-tumor images in 'no_tumor' folder)")
        print("4. Run the script again")
