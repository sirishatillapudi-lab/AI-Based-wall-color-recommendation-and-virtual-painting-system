from PIL import Image

from models.virtual_paint import detect_wall_mask


image = Image.open("test_room.png")

mask = detect_wall_mask(image)

print("Wall detection completed.")

print("Wall pixels detected:", (mask > 0).sum())