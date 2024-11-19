import numpy as np
from functions.code2axis import code2axis


def cal_area_c(chain_code,coordinates):
    # coordinates = code2axis(chain_code, [0, 0])
    # print(coordinates)
    min_y = min(coordinates[:, 1])
    max_y = max(coordinates[:, 1])
    area = 0
    for y in range(min_y, max_y + 1):
        intersection_indices = np.where(coordinates[:, 1] == y)[0]
        intersection_count = len(intersection_indices)
        intersections = np.zeros(intersection_count)
        for i, index in enumerate(intersection_indices):
            intersections[i] = coordinates[index, 0]
        intersections = np.sort(intersections)
        for i in range(0, intersection_count - 1, 2):
            area += (intersections[i+1] - intersections[i]) + 1

    print('area(pixel):', area)
    
    circumference = 0
    for code in chain_code:
        if code % 2 == 0:
            circumference += np.linalg.norm([1, 0])
        else:
            circumference += np.linalg.norm([1, 1])

    print('circumference(pixel):', circumference)
    return area, circumference


