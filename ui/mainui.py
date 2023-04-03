import tkinter as tk
import customtkinter as ctk
import prelocatorUI as plocUI

class mainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        self.title("Intersection finder")
        self.geometry("400x240")

        self.prelocatorWindow = None

        self.button = ctk.CTkButton(master=self, text="Prelocator", command=self.openPrelocatorWindow)
        self.button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def openPrelocatorWindow(self):
        if self.prelocatorWindow is None or not self.prelocatorWindow.winfo_exists():
            self.prelocatorWindow = plocUI.prelocatorWindow(self)
        else:
            self.prelocatorWindow.focus()
        

class imageFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.label = ctk.CTkLabel(self)

