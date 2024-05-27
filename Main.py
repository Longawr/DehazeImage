import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import Dehaze

class DehazeApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Dehaze App")
        self.geometry("800x600")
        self.resizable(True, True)

        # Main Frame
        self.main_frame = tk.Frame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Input Image Label
        self.input_img_label = tk.Label(self.main_frame)
        self.input_img_label.grid(row=0, column=0)

        # Output Image Label
        self.output_img_label = tk.Label(self.main_frame)

        # Sidebar Frame
        self.sidebar_frame = tk.Frame(self)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Execute Time Label
        self.execute_time_label = tk.Label(self.sidebar_frame, text="Execute Time: N/A")
        self.execute_time_label.grid(row=0, column=0, sticky="w")

        # PSNR Label
        self.psnr_label = tk.Label(self.sidebar_frame, text="PSNR: N/A")
        self.psnr_label.grid(row=1, column=0, sticky="w")

        # SSIM Label
        self.ssim_label = tk.Label(self.sidebar_frame, text="SSIM: N/A")
        self.ssim_label.grid(row=2, column=0, sticky="w")

        # Parameters Frame
        self.parameters_frame = tk.LabelFrame(self.sidebar_frame, text="Parameters")
        self.parameters_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        # Parameter Inputs
        self.tmin_entry = self.create_parameter_input("Minimum Transmission(0.1-0.5):", default=0.3)
        self.w_entry = self.create_parameter_input("Window Size(10 - 20):", default=15)
        self.alpha_entry = self.create_parameter_input("Alpha(0.1 - 0.5):", default=0.4)
        self.omega_entry = self.create_parameter_input("Omega(0.5 - 1):", default=0.75)
        self.p_entry = self.create_parameter_input("Percentage(1e-3 - 0.1):", default=0.1)
        self.eps_entry = self.create_parameter_input("Epsilon(1e-3 - 0.1):", default=1e-3)
        self.reduce_var = tk.BooleanVar(value=False)
        self.reduce_checkbutton = tk.Checkbutton(self.parameters_frame, text="Reduce Initial Transmission", variable=self.reduce_var)
        self.reduce_checkbutton.grid(row=12, column=0, columnspan=2, sticky="w")

        # Buttons Frame
        self.buttons_frame = tk.Frame(self.sidebar_frame)
        self.buttons_frame.grid(row=7, column=0, pady=10)

        # Open Image Button
        self.open_button = tk.Button(self.buttons_frame, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=5)

        # Run Dehaze Button
        self.run_button = tk.Button(self.buttons_frame, text="Run Dehaze", command=self.run_dehaze)
        self.run_button.grid(row=0, column=1, padx=5)

        # Save Image Button
        self.save_button = tk.Button(self.buttons_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=5)

    def create_parameter_input(self, label_text, default):
        label = tk.Label(self.parameters_frame, text=label_text)
        label.grid(sticky="w")

        entry = tk.Entry(self.parameters_frame)
        entry.grid(sticky="ew")
        entry.insert(0, str(default))  # Set default value
        return entry

    def open_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if filepath:
            # Get max frame dimensions (half of the main frame size)
            buffer = 90
            sidebar_width = 100
            max_frame_width = self.winfo_screenwidth() - buffer - sidebar_width
            max_frame_height = self.winfo_screenheight() - buffer
            print("Max Frame Width:", max_frame_width, "Max Frame Height:", max_frame_height)
            cv_image = Dehaze.open_image(filepath)
            # Resize image
            self.input_image = Dehaze.resize_image(cv_image, max_frame_width, max_frame_height)
            print(self.input_image.shape)

            input_image_rgb = Dehaze.bgr_to_rgb(self.input_image)
            input_image_display = ImageTk.PhotoImage(Image.fromarray(input_image_rgb))
            
            self.input_img_label.config(image=input_image_display)
            self.input_img_label.image = input_image_display  # Giữ tham chiếu để tránh bị garbage collected
            
            self.output_img_label.config(image=None)  # Clear output image
            self.update_window_size()

            self.clear_metrics()  # Clear metrics
            
    def run_dehaze(self):
        parameters = {
            "tmin": float(self.tmin_entry.get()),
            "w": int(self.w_entry.get()),
            "alpha": float(self.alpha_entry.get()),
            "omega": float(self.omega_entry.get()),
            "p": float(self.p_entry.get()),
            "eps": float(self.eps_entry.get()),
            "reduce": self.reduce_var.get()
        }

        if hasattr(self, "input_image"):
            output_image, execution_time = Dehaze.run_dehaze(self.input_image, **parameters)
            
            self.output_image_bgr = output_image  # Lưu ảnh BGR để có thể lưu lại
            output_image_rgb = Dehaze.bgr_to_rgb(output_image)  # Chuyển đổi sang RGB để hiển thị
            output_image_display = ImageTk.PhotoImage(Image.fromarray(output_image_rgb))

            self.output_img_label.config(image=output_image_display)
            self.output_img_label.image = output_image_display  # Giữ tham chiếu để tránh bị garbage collected
            if(self.input_image.shape[0] > self.input_image.shape[1]):
                self.output_img_label.grid(row=0, column=1)
            else:
                self.output_img_label.grid(row=1, column=0)

            self.update_window_size()

            psnr = Dehaze.calculate_psnr(self.input_image, output_image)
            ssim = Dehaze.calculate_ssim(self.input_image, output_image)

            self.execute_time_label.config(text=f"Execute Time: {execution_time:.2f} seconds")
            self.psnr_label.config(text=f"PSNR: {psnr:.2f}")
            self.ssim_label.config(text=f"SSIM: {ssim:.2f}")
            messagebox.showinfo("Dehaze image", "Image dehazed successfully")
        else:
            messagebox.showerror("Error", "Please open an image first.")

    def save_image(self):
        if hasattr(self, "output_image_bgr"):
            filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if filepath:
                Dehaze.save_image(self.output_image_bgr, filepath)
                messagebox.showinfo("Save Image", "Image saved successfully.")
        else:
            messagebox.showerror("Error", "No image to save.")

    def clear_metrics(self):
        self.execute_time_label.config(text="Execute Time: N/A")
        self.psnr_label.config(text="PSNR: N/A")
        self.ssim_label.config(text="SSIM: N/A")

    def update_window_size(self):
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
    
if __name__ == "__main__":
    app = DehazeApp()
    app.mainloop()