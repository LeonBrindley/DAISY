from PIL import Image
import math
from pillow_heif import register_heif_opener
import os
import uuid

def calculate_scaling_factor(original_width, original_height, target_width, target_height):
    original_area = original_width * original_height
    target_area = target_width * target_height
    return math.sqrt(target_area / original_area)



# Returns the unique ID of the output fragments, which are saved at output_dir/{unique_id}*.jpg
# Also returns an array of fragment paths.
def preprocess_and_split_image(image_path, output_dir, target_width=2048, target_height=1536, segment_size=224, stride=194):
    unique_id = uuid.uuid4()
    result_paths = []

    with Image.open(image_path) as img:
        original_width, original_height = img.size
        scaling_factor = calculate_scaling_factor(original_width, original_height, target_width, target_height)
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        img_width, img_height = img_resized.size

        for x in range(0, img_width, stride):
            for y in range(0, img_height, stride):
                # Calculate the boundaries of the segment
                adjusted_x = min(x, img_width - segment_size)
                adjusted_y = min(y, img_height - segment_size)

                # Create the segment
                segment = img_resized.crop((adjusted_x, adjusted_y, adjusted_x + segment_size, adjusted_y + segment_size))

                # Convert to RGB and save as BMP without metadata
                segment = segment.convert("RGB")
                segment_output_path = os.path.join(output_dir, f"{unique_id}_{adjusted_x}_{adjusted_y}.jpg")
                segment.save(segment_output_path, format='JPEG')

                result_paths.append(segment_output_path)

    unique_filename = f"{unique_id}*.jpg"
    print(f"Image {image_path} has been preprocessed and fragmented and saved as {os.path.join(output_dir, f'{unique_filename}')}")
    return unique_id, result_paths