import io
from PIL import Image
from taipy.gui import Gui
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Initial state variables
path = None
status = None 
original_image = None
magnitude_plot = None
reconstructed_image = None
original_size = 0
reconstructed_size = 0
spatial_filtered_image = None
frequency_filtered_image = None
spatial_filtered_size = 0
frequency_filtered_size = 0
filter_size = 3
freq_range = (0.0, 1.0)
show_filters = False
preview_matrix = None

# Initialize filter matrix with proper column names
initial_matrix = np.ones((3, 3))
filter_matrix = pd.DataFrame(
    initial_matrix,
    columns=[f"Col{i}" for i in range(initial_matrix.shape[1])],
    index=[f"Row{i}" for i in range(initial_matrix.shape[0])]
)

# Updated markdown
md = """
<|part|
<p style="font-family: Arial, sans-serif; font-size: 48px; font-weight: bold; text-align: center; background: linear-gradient(45deg, #FF5733, #FFC300, #FF5733); background-size: 200% auto; color: transparent; background-clip: text; -webkit-background-clip: text; animation: shine 3s linear infinite; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); padding: 10px; letter-spacing: 2px; margin: 20px 0;">SpectrumCraft</p>
|>

<|part|
<p style="font-family: Arial, sans-serif; font-size: 28px; text-align: center; background: linear-gradient(45deg, #0099CC, #33ccff); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); margin-top: 10px;">Custom Filters & Frequency Tuning</p>
|>

<|part|
<div style="max-width: 800px; margin: 30px auto; padding: 20px; background: rgba(26, 26, 26, 0.6); border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <p style="font-family: Arial, sans-serif; font-size: 16px; color: white; margin-bottom: 15px;">How to use:</p>
    <ol style="font-family: Arial, sans-serif; font-size: 14px; color: white; margin-left: 20px; line-height: 1.6;">
        <li>Upload an image using the file uploader</li>
        <li>Observe the original image, its spectrum, and reconstructed version</li>
        <li>Experiment with spatial domain filters by adjusting the filter matrix</li>
        <li>Try frequency domain filtering by setting custom bandpass ranges</li>
        <li>Compare the results and file sizes of processed images</li>
    </ol>
</div>
|>

<|layout|class_name=controls-section|
Upload an image file: <|{path}|file_selector|extensions=png,jpg,jpeg|hover_text="Upload an image file"|on_action=process_image|>   
<|{status}|status|>
|>

<|layout|columns=1 1 1|gap=30px|visible={original_image is not None}|
<|part|class_name=image-section|
## Original Image
<|{original_image}|image|>
<|text-center|
File size: <|{original_size}|text|format=%.2f|> KB
|>
|>

<|part|class_name=image-section|
## Magnitude Plot
<|{magnitude_plot}|image|>
|>

<|part|class_name=image-section|
## Reconstructed Image
<|{reconstructed_image}|image|>
<|text-center|
File size: <|{reconstructed_size}|text|format=%.2f|> KB
|>
|>
|>

<|layout|columns=1 1|gap=30px|class_name=filters-section|visible={show_filters}|
<|part|class_name=filter-block|
## Spatial Domain Filter
Filter Size:   
<|{filter_size}|slider|min=3|max=15|step=2|on_change=update_matrix|>

### Edit Filter Matrix
<|{filter_matrix}|table|editable=True|width=100%|show_all|rebuild|on_edit=handle_matrix_edit|>
<|{preview_matrix}|text|class_name=matrix-preview|>  
<|Apply Spatial Filter|button|on_action=apply_spatial_filter|class_name=action-button|>

<|layout|visible={spatial_filtered_image is not None}|
<|part|class_name=image-section|
<|{spatial_filtered_image}|image|>
<|text-center|
File size: <|{spatial_filtered_size}|text|format=%.2f|> KB
|>
|>
|>
|>

<|part|class_name=filter-block|
## Frequency Domain Filter
Frequency Range:  
<|{freq_range}|slider|min=0|max=1|step=0.01|>  
<|Apply Frequency Filter|button|on_action=apply_frequency_filter|class_name=action-button|>

<|layout|visible={frequency_filtered_image is not None}|
<|part|class_name=image-section|
<|{frequency_filtered_image}|image|>
<|text-center|
File size: <|{frequency_filtered_size}|text|format=%.2f|> KB
|>
|>
|>
|>
|>
"""


def validate_image(file_path):
    """Validate if the file is a proper image and has the correct format"""
    if file_path is None:
        return False, "Please select an image file"
    
    try:
        with Image.open(file_path) as img:
            if img.format not in ['JPEG', 'PNG']:
                return False, f"Invalid image format. Got {img.format}, expected JPEG or PNG"
            img.verify()
            return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def process_image(state):
    """Process image with validation"""
    # Reset everything
    state.show_filters = False
    state.magnitude_plot = None
    state.reconstructed_image = None
    state.original_image = None
    state.spatial_filtered_image = None
    state.frequency_filtered_image = None
    state.original_size = 0
    state.reconstructed_size = 0
    state.spatial_filtered_size = 0
    state.frequency_filtered_size = 0
    state.preview_matrix = format_matrix_preview(state.filter_matrix)
    
    if not state.path:
        state.status = ("error", "Please select an image file")
        return
        
    # Validate image type
    is_valid, message = validate_image(state.path)
    if not is_valid:
        state.status = ("error", message)
        return
        
    try:
        with Image.open(state.path) as image:
            # Convert to grayscale and process
            gray_image = image.convert('L')
            image_array = np.array(gray_image)
            
            # Save original image
            img_byte_arr = io.BytesIO()
            gray_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            state.original_image = img_byte_arr.getvalue()
            state.original_size = get_file_size(image_array)
            
            # Create magnitude plot
            f_shift, magnitude_plot_bytes = create_magnitude_plot(image_array)
            state.magnitude_plot = magnitude_plot_bytes
            
            # Create reconstructed image
            reconstructed = np.fft.ifft2(np.fft.ifftshift(f_shift))
            reconstructed = np.abs(reconstructed)
            reconstructed_pil = array_to_image(reconstructed)
            
            buf = io.BytesIO()
            reconstructed_pil.save(buf, format='PNG')
            buf.seek(0)
            
            state.reconstructed_image = buf.getvalue()
            state.reconstructed_size = get_file_size(np.array(reconstructed_pil))
            
            # Show filters after successful processing
            state.show_filters = True
            state.status = ("success", "Image processed successfully")
            
    except Exception as e:
        state.status = ("error", f"Error processing image: {str(e)}")

def handle_matrix_edit(state, var_name, payload):
    """Handle edits to the filter matrix with formatted preview"""
    try:
        # Get the new value from payload
        new_value = float(payload["value"])
        
        # Update the matrix value
        row_idx = int(payload["index"])
        col_name = payload["col"]
        
        # Update matrix
        state.filter_matrix.at[f"Row{row_idx}", col_name] = new_value
        
        # Update preview with formatted matrix
        state.preview_matrix = format_matrix_preview(state.filter_matrix)
        
    except ValueError:
        state.status = ("error", "Please enter a valid number")
    except Exception as e:
        state.status = ("error", f"Error updating matrix: {str(e)}")

def update_matrix(state):
    """Update the filter matrix when size changes"""
    new_size = state.filter_size
    new_matrix = np.ones((new_size, new_size))
    
    # Create DataFrame with proper column names and index
    state.filter_matrix = pd.DataFrame(
        new_matrix,
        columns=[f"Col{i}" for i in range(new_size)],
        index=[f"Row{i}" for i in range(new_size)]
    )
    
    # Update preview with formatted matrix
    state.preview_matrix = format_matrix_preview(state.filter_matrix)



def safe_normalize(image_array):
    """Safely normalize image array to prevent division by zero"""
    if image_array.size == 0:
        return np.zeros_like(image_array)
        
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    if min_val == max_val:
        return np.zeros_like(image_array)
        
    normalized = np.clip((image_array - min_val) / (max_val - min_val), 0, 1)
    return normalized

def array_to_image(array):
    """Safely convert numpy array to PIL Image"""
    normalized = safe_normalize(array)
    img_array = (normalized * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def get_file_size(image_array):
    """Calculate file size in KB"""
    with io.BytesIO() as output:
        Image.fromarray((image_array * 255).astype(np.uint8)).save(output, format="PNG")
        return len(output.getvalue()) / 1024

def create_magnitude_plot(image_array):
    """Create magnitude spectrum plot"""
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f_shift, buf.getvalue()



def apply_bandpass_filter(spectrum, low_freq, high_freq):
    """Apply frequency domain bandpass filter"""
    rows, cols = spectrum.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r_outer = int(high_freq * min(crow, ccol))
    r_inner = int(low_freq * min(crow, ccol))
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(
        ((x - center[0])**2 + (y - center[1])**2 >= r_inner**2),
        ((x - center[0])**2 + (y - center[1])**2 <= r_outer**2)
    )
    mask[mask_area] = 1
    return spectrum * mask

def format_matrix_preview(matrix):
    """Format matrix for pretty display"""
    if matrix is None:
        return ""
    
    formatted = "Current Matrix Values:\n"
    arr = matrix.to_numpy()
    for row in arr:
        formatted += "[" + " ".join(f"{val:6.2f}" for val in row) + "]\n"
    return formatted


def apply_spatial_filter(state):
    """Apply spatial domain filter"""
    if not state.show_filters:
        state.status = ("warning", "Please process an image first")
        return
        
    try:
        # Get the numerical values from the DataFrame
        filter_matrix = state.filter_matrix.to_numpy()
        
        if np.any(np.isnan(filter_matrix)):
            state.status = ("error", "Filter matrix contains invalid values")
            return
            
        # Normalize the filter
        filter_sum = np.sum(filter_matrix)
        if filter_sum != 0:
            filter_matrix = filter_matrix / filter_sum
        
        image_array = np.array(Image.open(io.BytesIO(state.original_image)).convert('L'))
        
        # Apply filter in frequency domain
        f_transform = np.fft.fft2(image_array)
        f_shift = np.fft.fftshift(f_transform)
        
        full_filter = np.zeros_like(image_array, dtype=float)
        center = np.array(full_filter.shape) // 2
        start = center - np.array(filter_matrix.shape) // 2
        end = start + np.array(filter_matrix.shape)
        full_filter[start[0]:end[0], start[1]:end[1]] = filter_matrix
        
        filtered_spectrum = f_shift * full_filter
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
        filtered_image = np.abs(filtered_image)
        
        # Convert and save filtered image
        image_pil = array_to_image(filtered_image)
        buf = io.BytesIO()
        image_pil.save(buf, format='PNG')
        buf.seek(0)
        
        state.spatial_filtered_image = buf.getvalue()
        state.spatial_filtered_size = get_file_size(np.array(image_pil))
        state.status = ("success", "Spatial filter applied successfully")
        
    except Exception as e:
        state.status = ("error", f"Error applying spatial filter: {str(e)}")


def apply_frequency_filter(state):
    """Apply frequency domain filter"""
    if not state.show_filters:
        state.status = ("warning", "Please process an image first")
        return
        
    try:
        low_freq, high_freq = state.freq_range
        
        if low_freq >= high_freq:
            state.status = ("error", "Low frequency must be less than high frequency")
            return
            
        image_array = np.array(Image.open(io.BytesIO(state.original_image)).convert('L'))
        f_transform = np.fft.fft2(image_array)
        f_shift = np.fft.fftshift(f_transform)
        
        filtered_spectrum = apply_bandpass_filter(f_shift, low_freq, high_freq)
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
        filtered_image = np.abs(filtered_image)
        
        image_pil = array_to_image(filtered_image)
        buf = io.BytesIO()
        image_pil.save(buf, format='PNG')
        buf.seek(0)
        
        state.frequency_filtered_image = buf.getvalue()
        state.frequency_filtered_size = get_file_size(np.array(image_pil))
        state.status = ("success", "Frequency filter applied successfully")
        
    except Exception as e:
        state.status = ("error", f"Error applying frequency filter: {str(e)}")

def upload(state):
    """Process image with validation"""
    # Reset state
    state.spatial_filtered_image = None
    state.frequency_filtered_image = None
    state.show_filters = False
    state.magnitude_plot = None
    state.reconstructed_image = None
    state.original_size = 0
    state.reconstructed_size = 0
    state.spatial_filtered_size = 0
    state.frequency_filtered_size = 0
    
    # Validate image
    is_valid, message = validate_image(state.path)
    if not is_valid:
        state.status = ("error", message)
        return
        
    try:
        with Image.open(state.path) as image:
            gray_image = image.convert('L')
            image_array = np.array(gray_image)
            
            img_byte_arr = io.BytesIO()
            gray_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            state.original_image = img_byte_arr.getvalue()
            state.original_size = get_file_size(image_array)
            
            f_shift, magnitude_plot_bytes = create_magnitude_plot(image_array)
            state.magnitude_plot = magnitude_plot_bytes
            
            reconstructed = np.fft.ifft2(np.fft.ifftshift(f_shift))
            reconstructed = np.abs(reconstructed)
            
            reconstructed_pil = array_to_image(reconstructed)
            buf = io.BytesIO()
            reconstructed_pil.save(buf, format='PNG')
            buf.seek(0)
            
            state.reconstructed_image = buf.getvalue()
            state.reconstructed_size = get_file_size(np.array(reconstructed_pil))
            
            state.show_filters = True
            state.status = ("success", "Image processed successfully")
            
    except Exception as e:
        state.status = ("error", f"Error processing image: {str(e)}")

if __name__ == "__main__":
    Gui(md).run(
    title="SpectrumCraft",
    port='auto',
    use_reloader=True  
)