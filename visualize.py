from PIL import Image, ImageDraw, ImageFont
import os
import json


def visualize_annotation(image_path, instruction, gt_bbox, pred_xy, save_path, img_size=None):
    """
    Visualize a single annotation on an image

    Args:
        image_path: Path to the image file
        instruction: Instruction text
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        save_path: Path to save the visualization
        img_size: Original image size for verification
    """
    # Create image object
    image = Image.open(image_path)

    # Verify image size if provided
    if img_size:
        assert image.size == (
            img_size[0],
            img_size[1],
        ), f"Image size mismatch: expected {img_size}, got {image.size}"

    # Create a copy to avoid modifying original
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            # Add more font paths if needed
        ]

        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 100)  # Increased font size to 40
                break

        if font is None:
            font = ImageFont.load_default(size=100)
    except Exception as e:
        print(f"Font loading failed: {e}")
        font = ImageFont.load_default(size=100)

    # Add instruction text with smart positioning
    text_color = (255, 0, 0)  # Red
    padding = 100  # Increased padding

    # Calculate text size
    text_bbox = draw.textbbox((0, 0), instruction, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Check if the text would overlap with the bounding box
    # First try top-left position
    text_position = (padding, padding)
    text_box = [
        text_position[0] - 100,
        text_position[1] - 100,
        text_position[0] + text_width + 100,
        text_position[1] + text_height + 100,
    ]

    # Check if text overlaps with annotation box
    def boxes_overlap(box1, box2):
        return not (
            box1[2] < box2[0]  # box1 is left of box2
            or box1[0] > box2[2]  # box1 is right of box2
            or box1[3] < box2[1]  # box1 is above box2
            or box1[1] > box2[3]
        )  # box1 is below box2

    # If text overlaps with annotation box, try alternative positions
    if boxes_overlap(text_box, gt_bbox):
        # Try top-right
        text_position = (image.width - text_width - padding, padding)
        text_box = [
            text_position[0] - 5,
            text_position[1] - 5,
            text_position[0] + text_width + 5,
            text_position[1] + text_height + 5,
        ]

        # If still overlapping, try bottom-left
        if boxes_overlap(text_box, gt_bbox):
            text_position = (padding, image.height - text_height - padding)
            text_box = [
                text_position[0] - 5,
                text_position[1] - 5,
                text_position[0] + text_width + 5,
                text_position[1] + text_height + 5,
            ]

            # If still overlapping, try bottom-right
            if boxes_overlap(text_box, gt_bbox):
                text_position = (
                    image.width - text_width - padding,
                    image.height - text_height - padding,
                )

    # Add white background to text for better readability
    text_box = [
        text_position[0] - 5,
        text_position[1] - 5,
        text_position[0] + text_width + 5,
        text_position[1] + text_height + 5,
    ]
    draw.rectangle(text_box, fill="white")
    draw.text(text_position, instruction, fill=text_color, font=font)

    # Draw bounding box in green
    annotation_color = (0, 255, 0)  # Green
    draw.rectangle(gt_bbox, outline=annotation_color, width=10)
    draw.circle(pred_xy, 10, fill="red")

    # Save the visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_draw.save(save_path)
    print(f"Saved visualization to {save_path}")


def process_predictions(predictions, output_dir="./visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    predictions = predictions["details"]

    for prediction in predictions:
        img_path = prediction["img_path"]
        instruction = prediction["prompt_to_evaluate"]
        gt_bbox = prediction["bbox"]
        pred_xy = prediction["pred"]
        group = prediction["group"]
        os.makedirs(os.path.join(output_dir, group), exist_ok=True)
        save_path = os.path.join(output_dir, group, f"{prediction['img_path'].split('/')[-1]}_visualized.png")
        visualize_annotation(img_path, instruction, gt_bbox, pred_xy, save_path)


if __name__ == "__main__":
    predictions = json.load(open("./results/aguvis_72b.json", "r"))
    process_predictions(predictions)
