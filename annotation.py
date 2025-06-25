import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from sadtalker_video2pose.src.face3d.extract_kp_videos_safe import KeypointExtractor

# annotation for input characters
image_dir = './characters/images/'
points_dir = './characters/points/'
visual_dir = './characters/visualizations/'
character_domains = os.listdir(image_dir)
for domain in character_domains:
    image_files = [f for f in os.listdir(os.path.join(image_dir, domain)) if f.endswith('.jpg')] 
    detector = KeypointExtractor()   

    for image_file in image_files:
        image_path = os.path.join(image_dir, domain, image_file)
        if os.path.exists(image_path[:-4]+".npy"):
            continue
        print(image_path)
        image = Image.open(image_path)
        image_np = np.array(image)
        # annotation for humans
        points = detector.extract_keypoint(image_np).astype(int)
        # mannual annotation for non-human characters
        if np.all(points == -1.):
            def onclick(event):
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    print(f"coordinates：({x}, {y})")
                    points.append([x, y])
                    ax.plot(x, y, 'ro')
                    fig.canvas.draw()
            points = []
            fig, ax = plt.subplots()
            ax.imshow(image_np)
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            points = np.array(points).astype(int)

        save_name = os.path.splitext(image_file)[0] + '.npy'
        save_points_dir = os.path.join(points_dir, domain)
        os.makedirs(save_points_dir, exist_ok=True)
        np.save(os.path.join(save_points_dir, save_name), points)
        print(f"saving landmarks in {save_points_dir}")

        img = cv2.imread(image_path)
        # plot on raw image
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(
                img,
                text=str(i+1),
                org=(x + 10, y + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )
        save_visual_dir = os.path.join(visual_dir, domain)
        os.makedirs(save_visual_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_visual_dir, save_name)[:-4]+"_visual.jpg", img)

    print(f"Process domain {domain} end.")