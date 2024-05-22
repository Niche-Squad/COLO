import numpy as np
import PIL

BATCH = 64

imgs = np.random.randint(0, 255, (BATCH, 640, 640, 3)).astype(np.uint8)
# export
for i, img in enumerate(imgs):
    img = PIL.Image.fromarray(img)
    # save
    img.save("image_%d.jpg" % i)
