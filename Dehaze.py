import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

from Modules import get_illumination_channel, get_atmosphere, get_initial_transmission, reduce_init_t, get_corrected_transmission, guided_filter, get_final_image

# Giá trị mặc định cho các tham số
DEFAULT_PARAMETERS = {
    "tmin": 0.3,
    "w": 15,
    "alpha": 0.4,
    "omega": 0.75,
    "p": 0.1,
    "eps": 1e-3,
    "reduce": False
}

def check_param(name, value, min_val, max_val, types):
    """
    Check the data type and value range of a parameter.

    Parameters:
    - name: Name of the parameter.
    - value: Value of the parameter.
    - min_val: Minimum value for the parameter.
    - max_val: Maximum value for the parameter.
    - types: Valid data types for the parameter.

    Raises:
    - TypeError if the value is not of the specified types.
    - ValueError if the value is out of the specified range.
    """
    if not isinstance(value, types):
        raise TypeError(f"{name} must be one of the types {types}.")
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in the range [{min_val}, {max_val}].")

def dehaze(I, tmin=0.3, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False):
    """
    This function enhances the visibility of an image affected by haze or fog.
    
    Parameters:
    - I: Input image.
    - tmin: Minimum threshold for transmission.
    - w: Window size for operations.
    - alpha: Adjustment parameter.
    - omega: Brightness adjustment parameter.
    - p: Percentage of the brightest pixels to estimate the environmental light.
    - eps: Adjustment parameter for the guided filter.
    - reduce: Reduce the initial transmission if True.
    
    Returns:
    - Enhanced image after dehazing.
    """
    # Kiểm tra kiểu dữ liệu và giá trị của ảnh đầu vào
    if not isinstance(I, np.ndarray):
        raise TypeError("I phải là một numpy array.")
    if I.ndim != 3 or I.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh RGB 3 kênh.")

    # Kiểm tra các biến đầu vào
    check_param("tmin", tmin, 0.1, 0.5, (int, float))
    check_param("w", w, 10, 20, int)
    check_param("alpha", alpha, 0.1, 0.5, (int, float))
    check_param("omega", omega, 0.5, 1, (int, float))
    check_param("p", p, 0.001, 0.1, (int, float))
    check_param("eps", eps, 1e-3, 1e-1, (int, float))
    if not isinstance(reduce, bool):
        raise TypeError("reduce phải là một giá trị boolean.")

    # print('tmin=', tmin, 'w=', w, 'alpha=', alpha, 'omega=', omega, 'p=', p, 'eps=', eps, 'reduce=', reduce)
    
    I = np.asarray(I, dtype=np.float64)  # Chuyển đổi đầu vào thành mảng số thực.
    I = I[:, :, :3] / 255  # Chuyển đổi giá trị màu RGB từ [0, 255] về [0, 1].

    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)
    init_t = get_initial_transmission(A, Ibright)
    
    if reduce:
        init_t = reduce_init_t(init_t)
        
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)
    normI = (I - I.min()) / (I.max() - I.min())  # Chuẩn hóa ảnh.
    refined_t = guided_filter(normI, corrected_t, w, eps)  # Áp dụng guided filter.
    J_refined = get_final_image(I, A, refined_t, tmin)
     
    enhanced = (J_refined * 255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

def calculate_psnr(original, processed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
    - original: Original image.
    - processed: Processed image.
    
    Returns:
    - PSNR value.
    """
    # Kiểm tra kiểu dữ liệu của ảnh đầu vào
    if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
        raise TypeError("original và processed phải là các mảng numpy.")

    # Kiểm tra kích thước của ảnh đầu vào
    if original.shape != processed.shape:
        raise ValueError("original và processed phải có cùng kích thước.")

    # Tính toán MSE (Mean Squared Error)
    mse = np.mean((original - processed) ** 2)

    # Tránh chia cho 0 bằng cách kiểm tra MSE
    if mse == 0:
        return float('inf')

    max_pixel = 255.0  # Giá trị tối đa của pixel
    psnr = 10 * np.log10((max_pixel ** 2) / mse)  # Tính toán PSNR
    return psnr

def calculate_ssim(original, processed):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
    - original: Original image.
    - processed: Processed image.
    
    Returns:
    - SSIM value.
    """
    # Check the data type of the input images
    if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
        raise TypeError("original and processed must be numpy arrays.")

    # Check the size of the input images
    if original.shape != processed.shape:
        raise ValueError("original and processed must have the same size.")

    # Calculate SSIM (Structural Similarity Index)
    ssim_value, _ = ssim(original, processed, full=True, channel_axis = 2)

    return ssim_value

def run_dehaze(I, tmin=None, w=None, alpha=None, omega=None, p=None, eps=None, reduce=None):
    """
    This function runs the dehaze function and measures the execution time.
    
    Parameters:
    - I: Input image.
    - Other parameters of the dehaze function.
    
    Returns:
    - Result of the dehaze function.
    - Execution time.
    """
    parameters = DEFAULT_PARAMETERS.copy()  # Sử dụng giá trị mặc định
    if tmin is not None:
        parameters["tmin"] = tmin
    if w is not None:
        parameters["w"] = w
    if alpha is not None:
        parameters["alpha"] = alpha
    if omega is not None:
        parameters["omega"] = omega
    if p is not None:
        parameters["p"] = p
    if eps is not None:
        parameters["eps"] = eps
    if reduce is not None:
        parameters["reduce"] = reduce

    # Thực hiện xử lý hình ảnh và đo thời gian thực thi
    start_time = time.time()
    result = dehaze(I, **parameters)
    end_time = time.time()
    execution_time = end_time - start_time

    return result, execution_time
    
def open_image(image_path):
    """
    Open and return the image from the specified path using OpenCV.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Loaded image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to load the image.")
        return None
    else:
        print("Image loaded successfully.")
        return image

def ensure_same_size(original_image, processed_image):
    """
    Ensure that processed_image has the same size with original_image by resizing them accordingly.

    Parameters:
    - original_image: original image.
    - processed_image: processed image.

    Returns:
    - resized processed_image.
    """
    if original_image.shape != processed_image.shape:
        processed_image = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))

    return processed_image

def bgr_to_rgb(image):
    """
    Convert an image from BGR to RGB format.

    Parameters:
    - image: The image in BGR format.

    Returns:
    - The image in RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def save_image(image, filename):
    """
    Save an image to a file using OpenCV.

    Parameters:
    - image: The image to be saved.
    - filename: The filename to save the image to.
    """
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

def show_image(image, window_name='Image'):
    """
    Display an image using OpenCV.

    Parameters:
    - image: The image to be displayed.
    - window_name: The name of the window in which the image will be displayed.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image, max_width, max_height):
    """
    Resize the image to fit within the maximum width and maximum height while maintaining aspect ratio.

    Parameters:
    - image: The input image.
    - max_width: The maximum width.
    - max_height: The maximum height.

    Returns:
    - Resized image.
    """
    height, width = image.shape[:2]

    # Check if the image is in landscape or portrait orientation
    is_landscape = width >= height

    if is_landscape:
        # Landscape orientation
        if width > max_width or height > max_height // 2:
            # Resize width to fit within max_width
            scale_factor = min(max_width / width, (max_height // 2) / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image
    else:
        # Portrait orientation
        if height > max_height or width > max_width // 2:
            # Resize height to fit within max_height
            scale_factor = min(max_height / height, (max_width // 2) / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image

    # Return the original image if no resizing is needed
    return image