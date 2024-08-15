import cv2
import math
import numpy as np

braille_dict = {
    'a': '100000',
    'b': '101000',
    'c': '110000',
    'd': '110100',
    'e': '100100',
    'f': '111000',
    'g': '111100',
    'h': '101100',
    'i': '011000',
    'j': '011100',
    'k': '100010',
    'l': '101010',
    'm': '110010',
    'n': '110110',
    'o': '100110',
    'p': '111010',
    'q': '111110',
    'r': '101110',
    's': '011010',
    't': '011110',
    'u': '100011',
    'v': '101011',
    'w': '011101',
    'x': '110011',
    'y': '110111',
    'z': '100111',
    ' ': '000000',
}


def find_bounding_boxes(label_matrix, obj_list):
    bounding_boxes = {}

    for label in range(len(obj_list)):
        # Find the rows and columns where the labeled region appears in the label matrix
        rows, cols = np.where(label_matrix == obj_list[label])

        # Calculate the minimum and maximum rows and columns
        min_row = np.min(rows)
        max_row = np.max(rows)
        min_col = np.min(cols)
        max_col = np.max(cols)
        x1 = (max_row + min_row) // 2
        y1 = (min_col + max_col) // 2
        bounding_boxes[label] = (x1, y1)

    return bounding_boxes


def cal_dis(mid_dot):
    p1 = mid_dot[0]
    p2 = mid_dot[1]
    p3 = mid_dot[2]
    dis1 = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    dis2 = int(math.hypot(p3[0] - p2[0], p3[1] - p2[1]))
    max_dis = int(dis1 + dis2)
    return dis1, max_dis, dis2


def count_object(my_array):
    my_array = np.array(my_array, dtype='uint16')
    m1 = np.pad(my_array, [(1, 1), (1, 1)], mode='constant', constant_values=0)  # padding top and left row of zeors
    obj_list = []
    size = np.shape(m1)  # size of img
    row = size[0]
    col = size[1]
    counter = 0
    label = np.zeros_like(m1)  # label matrix
    for i in range(1, row - 1):
        for j in range(1, col - 1):  # creating label matrix 1st iteration
            if m1[i][j] == 0:
                neighbors = [label[i - 1][j - 1], label[i - 1][j], label[i - 1][j + 1], label[i][j - 1]]  # checking
                # neighbours and giving the minimum value to pixel
                connected_neighbors = [x for x in neighbors if x > 0]
                if connected_neighbors:
                    min_label = min(connected_neighbors)
                    label[i][j] = min_label
                    for neighbor in connected_neighbors:
                        if neighbor != min_label:
                            obj_list[min_label - 1] += obj_list[neighbor - 1]  # creating list to see if the
                            # connected object has any other value
                            obj_list[neighbor - 1] = []  # clearing the overlying neighbours
                else:
                    counter += 1  # if no neighbour iterate
                    label[i][j] = counter
                    obj_list.append([counter])

    obj_list = [obj for obj in obj_list if obj]
    obj_counter = len(obj_list)
    print("number of objects are ", obj_counter)

    rep_labels = {}  # creating dictionary
    for elem in obj_list:
        if len(elem) > 1:
            # use the first element as the representative label for the object
            rep_labels.update({label_val: elem[0] for label_val in elem})

    # replace the labels in the label matrix using the representative labels
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            label_val = label[i][j]
            if label_val in rep_labels:
                label[i][j] = rep_labels[label_val]

    # cv2.imshow("label", label)
    # cv2.waitKey(0)
    cv2.imwrite('label_matrix.png', label)

    return label, obj_list


img = cv2.imread("Braille.png", 0)
size = np.shape(img)
new_size = (540, 700)
# Resize the image
interpol = cv2.INTER_AREA
gray = cv2.resize(img, new_size, interpolation=interpol)
size = np.shape(gray)
for i in range(size[0]):  # restricting it to 1 level
    for j in range(size[1]):
        if gray[i][j] <= 128:
            gray[i][j] = 0
        else:
            gray[i][j] = 255

label_matrix, obj_list = count_object(gray)
mid_dot = find_bounding_boxes(label_matrix, obj_list)

min_dis, max_dis, threshold_dis = cal_dis(mid_dot)

# set to keep track of unique values
unique_row = set()
unique_col = set()

# iterate through values in the dictionary
for value in mid_dot.values():
    # add first value in tuple to set
    unique_row.add(value[0])
    unique_col.add(value[1])
# convert set back to list
unique_row_list = sorted(list(unique_row))  # all the rows where the values start
unique_col_list = sorted(list(unique_col))
z1 = len(unique_row_list)
z2 = len(unique_col_list)
char_list = []

prev_key =0
range1 = (2 * min_dis) + 1
range2 = min_dis + 1
for l in range(0, z1, 3):  # iterating column by columnn
    k = unique_col_list[0]
    line_ended = False
    while True:
        my_list = []
        exist = True
        n = unique_row_list[l]
        o = k
        for m in range(n, n + range1, min_dis):
            for t in range(o, o + range2, min_dis):
                if label_matrix[m][t] != 0:
                    my_list.append(1)
                else:
                    my_list.append(0)

        binary_str = ''.join(map(str, my_list))
        if binary_str in braille_dict.values():
            for character, dots_str in braille_dict.items():
                if dots_str == binary_str:
                    char_list.append(character)
                    break
        if len(char_list) > 1:
            if char_list[-1] == ' ':
                k = k + max_dis + min_dis
                num1 = n
                matching_items = {i: v for i, v in mid_dot.items() if v[0] == num1}
                closest_key1 = min(matching_items, key=lambda i: abs(matching_items[i][1] - k))

                num1 = n + 5
                matching_items = {i: v for i, v in mid_dot.items() if v[0] == num1}
                closest_key2 = min(matching_items, key=lambda i: abs(matching_items[i][1] - k))

                num1 = n + 10
                matching_items = {i: v for i, v in mid_dot.items() if v[0] == num1}
                closest_key3 = min(matching_items, key=lambda i: abs(matching_items[i][1] - k))

                m1 = mid_dot[closest_key1]
                m2 = mid_dot[closest_key2]
                m3 = mid_dot[closest_key3]
                m = [k - m1[1], k - m2[1], k - m3[1]]
                min_val = min(x for x in m if x > 0)
                if min_val == k - m1[1]:
                    closest_m = m1
                elif min_val == k - m2[1]:
                    closest_m = m2
                else:
                    closest_m = m3

                k=int(closest_m[1])
                if prev_key != k:
                    prev_key=k
                else:
                    line_ended = False
                    break

            else:
                k = k + max_dis
            if k > 537:
                break
        else:
            k = k + max_dis

        if line_ended:
            del char_list[-1]
            break


decode_str = ''.join(map(str, char_list))
print(decode_str)
with open('outputfile2.txt', 'w') as f:
    # Write the string to the file
    f.write(decode_str)