from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation
from safetensors.torch import load_file
import argparse
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Remove background from an image')
    parser.add_argument('-i', '--image_path', required=True,
                        help='Path to input image')
    args = parser.parse_args()
    # Expand the ~ to full home directory path
    args.image_path = str(Path(args.image_path).expanduser().resolve())
    return args


def main():
    start_time = time.time()

    args = parse_args()

    # Get input path and create output path
    input_path = Path(args.image_path)
    # Force PNG extension for output
    output_path = input_path.parent / f"{input_path.stem}_no_bg.png"

    # Settings
    model_directory = "./"  # Directory containing config.json and model.safetensors
    model_weights = f"{model_directory}/model.safetensors"  # Full path to model.safetensors

    # use cuda device if available, otherwise default to cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load model
        print(f"Loading model on {device}...")
        config = AutoConfig.from_pretrained(model_directory, trust_remote_code=True)
        model = AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
        state_dict = load_file(model_weights)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Set up image transformation.
        # for model processing, it will resize a copy of the image to 1024x1024
        # this is because the model was trained on images of this size
        model_input_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load and preprocess image
        # model will output a mask at 1024x1024
        print("Processing image...")
        image = Image.open(input_path).convert("RGB")
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        # Generate mask (1024x1024)
        with torch.no_grad():
            predictions = model(input_tensor)[-1].sigmoid().cpu()

        # Process mask and apply to image...

        # Removes extra dimensions from tensor shape
        pred_mask = predictions[0].squeeze()
        # Convert tensor to PIL Image
        mask_pil = transforms.ToPILImage()(pred_mask)
        # mask is resized back to original dimensions
        mask_resized = mask_pil.resize(image.size)
        # full-size mask is applied to original image
        image.putalpha(mask_resized)

        # Save result as PNG
        image.save(output_path, 'PNG')

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Successfully saved image to {output_path}")
        print(f"Total processing time: {processing_time:.2f} seconds")

    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
    except torch.cuda.OutOfMemoryError:
        print("Error: Not enough GPU memory. Try using CPU device instead.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()