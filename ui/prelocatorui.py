import tkinter as tk
import customtkinter as ctk


class prelocatorWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300")

        self.title("Prelocator")

        self.label = ctk.CTkLabel(self, text="Prelocator")
        self.label.pack(padx=20, pady=20)

