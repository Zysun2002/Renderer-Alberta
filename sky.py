from PIL import Image
import trimesh

tga = Image.open("/root/autodl-tmp/tiny-renderer/assets/bob.tga")

tga.save("bob.png", "PNG")