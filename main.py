from GUI import GestureApp
import tkinter as tk
import customtkinter as ctk

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = GestureApp(root)
    root.mainloop()
