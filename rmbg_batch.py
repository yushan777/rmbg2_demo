from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation
from safetensors.torch import load_file
import argparse
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Remove background from images in a directory')
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Path to directory containing images')
    parser.add_argument('-o', '--output_dir', required=False,
                        help='Path to output directory (optional, defaults to input_dir/no_bg)')
    args = parser.parse_args()

    # Expand paths
    args.input_dir = str(Path(args.input_dir).expanduser().resolve())
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    else:
        # Default output directory is 'no_bg' subdirectory in input directory
        args.output_dir = str(Path(args.input_dir) / 'no_bg')

    return args


def process_image(image_path, output_path, model, transform_image, device):
    """Process a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        # Generate mask (1024x1024)
        with torch.no_grad():
            predictions = model(input_tensor)[-1].sigmoid().cpu()

        # Process mask and apply to image
        pred_mask = predictions[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred_mask)
        mask_resized = mask_pil.resize(image.size)
        image.putalpha(mask_resized)

        # Save result as PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, 'PNG')
        print(f"Processed: {image_path.name} -> {output_path.name}")
        return True
    except Exception as e:
        print(f"Error processing {image_path.name}: {str(e)}")
        return False


def main():
    start_time = time.time()
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of images to process
    input_dir = Path(args.input_dir)
    image_paths = []
    # Look for common image formats
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(input_dir.glob(ext))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Settings
    model_directory = "./"
    model_weights = f"{model_directory}/model.safetensors"
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

        # Set up image transformation
        model_input_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Process each image
        successful = 0
        failed = 0
        for idx, image_path in enumerate(image_paths, 1):
            print(f"\nProcessing image {idx}/{len(image_paths)}")
            # Always output as PNG
            output_path = output_dir / f"{image_path.stem}_no_bg.png"

            if process_image(image_path, output_path, model, transform_image, device):
                successful += 1
            else:
                failed += 1

        # Print summary
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nProcessing complete!")
        print(f"Total time: {processing_time:.2f} seconds")
        print(f"Average time per image: {processing_time / len(image_paths):.2f} seconds")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()