import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Modeli yükle
model = tf.keras.models.load_model("model.h5")


class RakamTanimaUygulamasi:
    def __init__(self, pencere):
        self.pencere = pencere
        self.pencere.title("El Yazısı ile Rakam Tanıma")

        self.tuval = tk.Canvas(pencere, width=400, height=400, bg="white")
        self.tuval.pack()

        self.buton_onayla = tk.Button(pencere, text="Onayla", command=self.rakami_tani)
        self.buton_onayla.pack()

        self.buton_temizle = tk.Button(
            pencere, text="Temizle", command=self.tuvali_temizle
        )
        self.buton_temizle.pack()

        self.tuval.bind("<B1-Motion>", self.ciz)
        self.tuval.bind("<ButtonPress-1>", self.cizime_basla)
        self.tuval.bind("<ButtonRelease-1>", self.cizimi_bitir)

        self.resim = Image.new("L", (400, 400), 255)
        self.cizim = ImageDraw.Draw(self.resim)

    def cizime_basla(self, olay):
        self.cizim_yapiliyor = True
        self.ciz(olay)

    def cizimi_bitir(self, olay):
        self.cizim_yapiliyor = False

    def ciz(self, olay):
        if not hasattr(self, "cizim_yapiliyor") or not self.cizim_yapiliyor:
            return
        x1, y1 = (olay.x - 5), (olay.y - 5)
        x2, y2 = (olay.x + 5), (olay.y + 5)
        self.tuval.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.cizim.ellipse([x1, y1, x2, y2], fill="black")

    def rakami_tani(self):
        self.resim = self.resim.resize((28, 28))
        self.resim = ImageOps.invert(self.resim)
        resim_dizisi = np.array(self.resim).reshape(1, 28, 28, 1) / 255.0

        tahmin = model.predict(resim_dizisi)
        rakam = np.argmax(tahmin)

        if np.max(tahmin) > 0.5:
            messagebox.showinfo("Sonuç", f"Tanınan Rakam: {rakam}")
        else:
            messagebox.showwarning("Uyarı", "Rakam tanınamadı. Lütfen tekrar deneyin.")

    def tuvali_temizle(self):
        self.tuval.delete("all")
        self.resim = Image.new("L", (400, 400), 255)
        self.cizim = ImageDraw.Draw(self.resim)


if __name__ == "__main__":
    pencere = tk.Tk()
    uygulama = RakamTanimaUygulamasi(pencere)
    pencere.mainloop()
