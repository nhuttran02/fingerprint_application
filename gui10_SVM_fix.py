import subprocess
import time
import os
from pathlib import Path
from tkinter import Tk, Canvas, filedialog
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue
from threading import Thread
import logging

#set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fingerprint_system.log'
)

#define temporary file paths
temp_image_path = Path(os.path.join(os.getenv('TEMP'), "temp_fingerprint.jpg"))  
temp_csv_path = Path(os.path.join(os.getenv('TEMP'), "temp_embedding.csv"))

#define paths and assets
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"Q:\Thay_Nghi\NCKH\Tkinter-Designer\build\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

#setup InceptionV3 model to embedding image from capture.cs
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

class FingerprintHandler(FileSystemEventHandler):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.last_processed_time = 0
        self.processing_lock = False
        self.processing_queue = Queue()
        self.start_processing_thread()

    def start_processing_thread(self):
        def process_queue():
            while True:
                try:
                    image_path = self.processing_queue.get()
                    if image_path is None:  # Signal to stop thread
                        break
                    
                    logging.info(f"Processing fingerprint from {image_path}")
                    embedding = embed_image(image_path)
                    compare_embedding_with_database(embedding)
                    
                    # Update UI in main thread
                    self.canvas.after(0, lambda: load_and_display_image(image_path))
                    
                except Exception as e:
                    logging.error(f"Error processing fingerprint: {e}")
                    self.canvas.after(0, lambda: self.canvas.itemconfig(
                        error_text, text=f"Error: {str(e)}")
                    )
                finally:
                    self.processing_queue.task_done()

        self.processing_thread = Thread(target=process_queue, daemon=True)
        self.processing_thread.start()

    def on_modified(self, event):
        if not event.is_directory and event.src_path == str(temp_image_path):
            current_time = time.time()
            if current_time - self.last_processed_time > 1 and not self.processing_lock:
                self.processing_lock = True
                try:
                    time.sleep(0.5)  # Wait for file to be completely written
                    self.processing_queue.put(str(temp_image_path))
                    self.last_processed_time = current_time
                finally:
                    self.processing_lock = False

    def stop(self):
        self.processing_queue.put(None)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

#define embedding function
def embed_image(image_path):   
    try:
        img = load_img(image_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = inception_model.predict(img_array)
        feature = feature.flatten()

        np.savetxt(temp_csv_path, feature, delimiter=',')
        return feature
    except Exception as e:
        logging.error(f"Error in embed_image: {e}")
        raise

#compare
def compare_embedding_with_database(embedding):
    try:
        database_csv = 'db_fp.csv'
        df_database = pd.read_csv(database_csv, header=None)

        X_database = df_database.iloc[:, :-1].values
        y_database = df_database.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_database, y_database, test_size=0.2, random_state=42
        )

        model = svm.SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)

        predicted_proba = model.predict_proba(embedding.reshape(1, -1))
        max_proba = np.max(predicted_proba)
        predicted_label = model.predict(embedding.reshape(1, -1))[0]

        threshold = 0.95

        if max_proba >= threshold:
            logging.info(f'Match found: {predicted_label} with confidence {max_proba:.2f}')
            
            info_csv = 'real.csv'
            df_info = pd.read_csv(info_csv)
            info_row = df_info[df_info['label'] == predicted_label]

            if not info_row.empty:
                update_ui_with_info(info_row)
            else:
                canvas.itemconfig(error_text, text="Label not found in database")
                clear_ui_info()
        else:
            logging.info("No match found")
            canvas.itemconfig(error_text, text="Fingerprint mismatch")
            clear_ui_info()

    except Exception as e:
        logging.error(f"Error in compare_embedding_with_database: {e}")
        raise

#display info function to GUI
def update_ui_with_info(info_row):
    canvas.itemconfig(name_text, text=info_row['name'].values[0])
    canvas.itemconfig(class_text, text=info_row['class'].values[0])
    canvas.itemconfig(birth_text, text=info_row['birth'].values[0])
    canvas.itemconfig(address_text, text=info_row['address'].values[0])
    canvas.itemconfig(error_text, text="")
    canvas.itemconfig(success_text, text="Successfully!")

#clear function when get unknown fingerprint
def clear_ui_info():
    canvas.itemconfig(success_text, text="")
    canvas.itemconfig(name_text, text="")
    canvas.itemconfig(class_text, text="")
    canvas.itemconfig(birth_text, text="")
    canvas.itemconfig(address_text, text="")

#show image from capture.cs to frame in GUI
def load_and_display_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((234, 234))
        imgg = ImageTk.PhotoImage(image)
        
        canvas.image = imgg
        canvas.create_image(63.0, 99.0, anchor="nw", image=imgg)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        canvas.itemconfig(error_text, text="Error loading image")

def start_fingerprint_monitoring():
    event_handler = FingerprintHandler(canvas)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(temp_image_path), recursive=False)
    observer.start()
    return observer

def capture_fingerprint_with_usb():
    try:
        # Start the capture application
        subprocess.Popen(['D:/DigitalPersona/U.are.U SDK/Windows/Bin/Debug/x64/UareUSampleCSharp_CaptureOnly.exe'])
        
        # Start the monitoring system
        observer = start_fingerprint_monitoring()
        
        # Store the observer in the window to stop it when closing
        window.observer = observer
    except Exception as e:
        logging.error(f"Error capturing fingerprint: {e}")
        canvas.itemconfig(error_text, text="Error capturing fingerprint")

def on_closing():
    if hasattr(window, 'observer'):
        window.observer.stop()
        window.observer.join()
    clean_up_temp_files()
    window.destroy()

def clean_up_temp_files():
    try:
        os.remove(temp_image_path)
        print(f"Temporary image {temp_image_path} deleted.")
        os.remove(temp_csv_path)
        print(f"Temporary CSV {temp_csv_path} deleted.")
    except FileNotFoundError:
        pass

#frame app
window = Tk()
window.title("Fingerprint Recognition System")
window.geometry("1000x550")
window.configure(bg = "#CCBABA")

#interface size
canvas = Canvas(window, bg="#F1F5F8", height=550, width=1000, bd=0, highlightthickness=0, relief="ridge")  
canvas.place(x=0, y=0)


#title
canvas.create_rectangle(0.0, 3.0, 1000.0, 74.0, fill="#5850e9", outline="")
display = canvas.create_rectangle(63.0, 99.0, 297.0, 333.0, fill="#FAFAFA", outline="")
canvas.create_text(63.0, 7.0, anchor="nw", text="FINGERPRINT RECOGNITION SYSTEM", fill="#FAFAFA", font=("Inter", 48 * -1))

#result rectangle
canvas.create_text(455.0, 176.0, anchor="nw", text="RESULT", fill="#000000", font=("Inter", 24 * -1))
submit_button = canvas.create_rectangle(419.0, 165.0, 581.0, 213.0, fill="#908AFF", outline="")

#input image IMG rectangle
canvas.create_text(152.0, 459.0, anchor="nw", text="IMG", fill="#000000", font=("Inter", 24 * -1))
input_button_img = canvas.create_rectangle(63.0, 449.0, 297.0, 499.0, fill="#908AFF", outline="")
canvas.tag_bind(input_button_img, "<Button-1>", lambda e: capture_fingerprint_with_usb())


canvas.create_text(500.0, 294.0, anchor="nw", text="Name:", fill="#000000", font=("Inter", 24 * -1))
canvas.create_text(500.0, 349.0, anchor="nw", text="Class", fill="#000000", font=("Inter", 24 * -1))
canvas.create_text(500.0, 402.0, anchor="nw", text="Birthday:", fill="#000000", font=("Inter", 24 * -1))
canvas.create_text(500.0, 457.0, anchor="nw", text="Address:", fill="#000000", font=("Inter", 24 * -1))

# Text elements for name, class, birth, and address
name_text = canvas.create_text(631.0, 289.0, anchor="nw", text="", fill="#000000", font=("Inter", 24 * -1))
class_text = canvas.create_text(631.0, 344.0, anchor="nw", text="", fill="#000000", font=("Inter", 24 * -1))
birth_text = canvas.create_text(631.0, 399.0, anchor="nw", text="", fill="#000000", font=("Inter", 24 * -1))
address_text = canvas.create_text(631.0, 454.0, anchor="nw", text="", fill="#000000", font=("Inter", 24 * -1))

#display error UI
error_text = canvas.create_text(631.0, 177.0, anchor="nw", text="", fill="red", font=("Inter", 24 * -1))

#display success UI
success_text = canvas.create_text(631.0, 177.0, anchor="nw", text="", fill="green", font=("Inter", 24 * -1))

window.protocol("WM_DELETE_WINDOW", on_closing)

window.resizable(False, False)
window.mainloop()