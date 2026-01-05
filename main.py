import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from sklearn.model_selection import cross_val_score
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np


# =========================
# create dataset
# =========================
# dataset_path = r"dataset"

# def load_and_process_image(path):
#     img = Image.open(path).convert("RGBA")
#     background = Image.new("RGBA", img.size, (255, 255, 255))
#     final = Image.alpha_composite(background, img).convert("L")  # grayscale


#     arr = np.array(final, dtype=np.uint8)
#     return arr

# def load_dataset(dataset_path):
#     images = []
#     labels = []


#     for digit in os.listdir(dataset_path):
#         digit_path = os.path.join(dataset_path, digit)
#         if os.path.isdir(digit_path):
#             for subfolder in os.listdir(digit_path):
#                 subfolder_path = os.path.join(digit_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".png")]
#                     for fname in tqdm(files, desc=f"Processing {digit}", unit="img"):
#                         fpath = os.path.join(subfolder_path, fname)
#                         try:
#                             arr = load_and_process_image(fpath)
#                             images.append(arr)
#                             labels.append(int(digit))
#                         except Exception as e:
#                             print("error in reading: ", fpath, e)

#     return np.array(images), np.array(labels)

# X, y = load_dataset(dataset_path)

# df = pd.DataFrame(X, columns=[f"pixel{i}" for i in range(784)])
# df["label"] = y


# df.to_csv("digits_dataset_raw_shuffled.csv", index=False)



# =========================
# Dataset Quality Check
# =========================

# df = pd.read_csv("digits_dataset_raw_shuffled.csv")

# X = df.drop(columns=["label"]).values   # (N, 784)
# y = df["label"].values                  # (N,)

# print(X.shape, y.shape)
# print(X.dtype, y.dtype)

# تست اول: نمایش سطر دلخواه
# idx = 40581
# img = X[idx].reshape(28, 28)
# label = y[idx]

# plt.imshow(img, cmap="gray")
# plt.title(f"Label = {label}")
# plt.axis("off")
# plt.show()


# تست دوم: نمایش چند سطر رندوم
# np.random.seed(42)
# indices = np.random.choice(len(X), 9, replace=False)

# plt.figure(figsize=(6, 6))
# for i, idx in enumerate(indices):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(X[idx].reshape(28, 28), cmap="gray")
#     plt.title(y[idx])
#     plt.axis("off")

# plt.tight_layout()
# plt.show()


# تست سوم: بازه مقادیر پیکسل‌ها
# print("min pixel:", X.min())
# print("max pixel:", X.max())


# تست چهارم: بالانس کلاس‌ها
# counter = collections.Counter(y)
# for k in sorted(counter.keys()):
#     print(f"Digit {k}: {counter[k]}")


# تست پنجم : درصد تصاویر تقریباً خالی
# white_ratio = (X > 250).mean(axis=1)

# print("Samples with >95% white pixels:",
#       np.sum(white_ratio > 0.95))

# bad_idxs = np.where(white_ratio > 0.95)[0]

# np.random.seed(12)
# sample = np.random.choice(bad_idxs, 25, replace=False)

# plt.figure(figsize=(10,10))
# for i, idx in enumerate(sample):
#     plt.subplot(5,5,i+1)
#     plt.imshow(X[idx].reshape(28,28), cmap="gray")
#     plt.title(f"label={y[idx]}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()


# bad_mask = white_ratio > 0.95
# labels_bad = y[bad_mask]

# counter = collections.Counter(labels_bad)

# for digit in range(10):
#     print(f"Digit {digit}: {counter.get(digit, 0)}")


# =========================
# Modeling
# =========================
# data = pd.read_csv("digits_dataset_raw_shuffled.csv")

# X = data.drop(columns=["label"])
# y = data["label"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = xg.XGBClassifier(
#     n_estimators=150,
#     max_depth=6,
#     learning_rate=0.1,
#     n_jobs=-1,
#     verbosity=2,
# )

# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_test, y_test)],
#     verbose=True,
# )
# joblib.dump(model,"model.pkl")

model = joblib.load("model.pkl")

##-- model quality
# y_pred = model.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# cr = classification_report(y_test, y_pred)
# print(cr)

# print("Train Accuracy:", model.score(X_train, y_train))
# print("Test Accuracy:", model.score(X_test, y_test))


##--  model test 1

# scores = cross_val_score(model, X[:10000], y[:10000], cv=5, n_jobs=-1)
# print(scores.mean(), scores.std())


##--  model test 2
# X_small = X_train[:10000]
# y_small = y_train[:10000]


# y_shuffled = np.random.permutation(y_small)


# model.fit(X_small, y_shuffled)

# print("Sanity check accuracy:", model.score(X_test, y_test))


# =========================
# GUI
# =========================


def preprocess_image(img):
    img = img.convert("L")
    img = np.array(img)

    img = cv2.resize(img, (28, 28))

    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    img = img.astype(np.uint8)


    img = img.reshape(1, -1)

    # نمایش انچه مدل میبیند 
    # plt.imshow(img.reshape(28, 28), cmap="gray")
    # plt.title("Final input to model")
    # plt.show()

    return img


def predict_digit():
    global uploaded_image
    if uploaded_image is None:
        result_label.config(text="لطفا ابتدا عکس آپلود کنید")
        return

    processed = preprocess_image(uploaded_image)
    pred = model.predict(processed)[0]
    result_label.config(text=f"عدد حدس زده شده: {pred}")


uploaded_image = None


def upload_image():
    global uploaded_image

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        img = Image.open(file_path)

        uploaded_image = img.copy()

        img_display = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_display)

        image_label.config(image=img_tk)
        image_label.image = img_tk


def reset_all():
    global uploaded_image
    uploaded_image = None

    image_label.config(image="")
    image_label.image = None

    result_label.config(text="")

##-- GUI panel
window = tk.Tk()
window.title("حدس اعداد انگلیسی دست نویس")
window.geometry("600x720")

window.grid_columnconfigure(0, weight=1)

label1 = tk.Label(window, text="لطفا عکس دست نوشته خود را آپلود کنید", font=("Arial", 14))
label1.grid(row=0, column=0, pady=(10, 0), sticky="n")


upload_button = tk.Button(window, text="آپلود عکس", command=upload_image, font=("Arial", 12))
upload_button.grid(row=1, column=0, pady=(10, 0))


image_label = tk.Label(window)
image_label.grid(row=2, column=0, pady=(10, 0))


predict_button = tk.Button(window, text="حدس عدد", command=predict_digit, font=("Arial", 12))
predict_button.grid(row=3, column=0, pady=(10, 10))

reset_button = tk.Button(window, text="ریست", command=reset_all, font=("Arial", 12))
reset_button.grid(row=4, column=0, pady=(5, 10))


result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.grid(row=5, column=0, pady=(10, 0))

window.mainloop()
