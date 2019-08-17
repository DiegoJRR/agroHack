import numpy as np
import cv2 as cv
import math
import random


def mapv(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def generate_heatmap(shape, min_v, max_v):
    # img = np.zeros((512,512,3), np.uint8)
    sensors = [
        [0, 0, {'moisture' : random.randint(min_v, max_v)}],
        [122, 112, {'moisture' : random.randint(min_v, max_v)}],
        [241, 41,{'moisture' : random.randint(min_v, max_v)}],
        [92, 321,{'moisture' : random.randint(min_v, max_v)}],
        [143, 312, {'moisture' : random.randint(min_v, max_v)}],
        [342, 232, {'moisture' : random.randint(min_v, max_v)}],
        [182, 197, {'moisture' : random.randint(min_v, max_v)}],
        [123, 48,{'moisture' : random.randint(min_v, max_v)}],
        [413, 482,{'moisture' : random.randint(min_v, max_v)}],
        [231, 102,{'moisture' : random.randint(min_v, max_v)}],
        [291, 392,{'moisture' : random.randint(min_v, max_v)}],
    ]
    value_map = []
    for i in range(shape[0]):
        li = []
        for j in range(shape[1]):
            li.append([0, 0])
        value_map.append(li)

    for i in range(shape[0]):
        for j in range(shape[1]):
            total_r = sum([get_dist(sensor[1], sensor[0], j, i) for sensor in sensors])
            for sensor in sensors:
                r = get_dist(sensor[1], sensor[0], i, j) 
                if (r > 0):
                    x = sensor[2]['moisture'] * (r / total_r)                
                    value_map[i][j][0] += x 
                    value_map[i][j][1] += 1

    
    # for i in range(512):
    #     for j in range(512):
    #         if (value_map[i][j][1] == 0):
    #             img[i][j][0] = 0
    #         else:
    #             img[i][j][2] = value_map[i][j][0] / value_map[i][j][1]
    #             img[i][j][2] = mapvimg[i][j][0], 0, value_map[i][j][0], 255, 0)

    sensor_data = np.array([[value_map[i][j][0] for j in range(shape[1])] for i in range(shape[0])])
    heatmapshow = None
    heatmapshow = cv.normalize(-sensor_data, heatmapshow, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmapshow = cv.applyColorMap(heatmapshow, cv.COLORMAP_JET)

    # result = cv.addWeighted(heatmapshow, 0.9, field, 0.1,0.0)
    # cv.imshow('image',heatmapshow)
    # cv.waitKey(0)

    return heatmapshow, sensor_data

# field = cv.resize(field, dsize=(512, 512))


# def loop():
#     value_map = fill_image()
#     imagen = np.array([[value_map[i][j][0] for j in range(img.shape[1])] for i in range(img.shape[0])])
#     heatmapshow = None
#     heatmapshow = cv.normalize(-imagen, heatmapshow, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
#     heatmapshow = cv.applyColorMap(heatmapshow, cv.COLORMAP_JET)
#     # cv.destroyAllWindows()

# result = cv.addWeighted(heatmapshow, 0.9, field, 0.1,0.0)
# cv.imshow('image',result)
# cv.waitKey(0)
#     # update_sensors()
# pass

# loop()
# img = generate_heatmap((512, 600), 0, 100)

# cv.imshow('image',img)
# cv.waitKey(0)