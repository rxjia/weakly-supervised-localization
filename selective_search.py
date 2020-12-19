import selectivesearch
import numpy as np
import skimage.data


def customized_selective_search(img):
    assert img.shape[-1] == 3, "imgs.shape should be [H, W, 3]"
    ss_boxes = []
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        if r['size'] < 50:
            continue
        x, y, w, h = r['rect']
        if w < 5 or h < 5 or w / h > 2.0 or h / w > 2.0:
            continue
        candidates.add(r['rect'])

    data = [list(a) for a in candidates]
    data = np.array(data)
    if data.shape[0] == 0:
        print("this image has no proposals")
    else:
        data[:, 2:] = data[:, :2] + data[:, 2:]
    ss_boxes.append(list(data))
    return ss_boxes


if __name__ == "__main__":
    img = skimage.data.astronaut()
    imgs = np.array([img])
    print(imgs.shape)
    ss_boxes = customized_selective_search(imgs, region_key='bbox')
    print(ss_boxes)