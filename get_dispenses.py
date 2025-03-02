# In[1]:


import cv2
import csv
import os
import json
import shutil
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from time import time
from PIL import Image, ImageDraw, ImageFont

from Detectron.predict_onnx import GetUpToDateModel as GetUpToDateDetectronModel
from Detectron.configuration import Configuration


# In[3]:


''' Downloading model '''

def download_model_from_url(deployment_server_url, model_name, model_stage):
    config = Configuration()
    config.deployment_server_url = deployment_server_url
    config.model_name = model_name
    config.model_stage = model_stage
    print(config.deployment_server_url)

    model = GetUpToDateDetectronModel("./Detectron/models", deployment_server_url, "-1",
                                      "station_name", model_name, model_stage,
                                      device="cuda")

    if model is None:
        raise SystemExit('Model was not loaded')

    return model


# In[4]:


''' Functions for finding the dispenses in each image '''

def extract_circle(image_gray, x, y, r):
    # Create a mask with the same dimensions as the image, initially filled with zeros (black)
    mask = np.zeros_like(image_gray)

    # Draw a white filled circle on the mask
    cv2.circle(mask, (x, y), r, 255, -1)

    # Extract the circular region from the image using the mask
    circle_image = cv2.bitwise_and(image_gray, mask)

    return circle_image


def plot_image_and_histogram(image_gray, x, y, r):
    color_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    circle_image = extract_circle(image_gray, x, y, r)

    # Calculate the histogram for the circular region
    hist, bins = np.histogram(circle_image[circle_image > 0].flatten(), bins=256, range=[0, 256])

    # Calculate mean and standard deviation
    mean_val = np.mean(circle_image[circle_image > 0])
    std_val = np.std(circle_image[circle_image > 0])

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    cv2.circle(color_image, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(color_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Display the original image with the circle highlighted on the left
    ax[0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('Original Image with Circle')

    # Display the histogram on the right
    ax[1].plot(hist, color='black')
    ax[1].set_xlim([0, 256])
    ax[1].set_title('Grayscale Histogram')

    # Draw mean value as a red vertical line
    ax[1].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')

    # Draw 2 standard deviation lines as orange vertical lines
    ax[1].axvline(mean_val - 2 * std_val, color='orange', linestyle='dashed', linewidth=1,
                  label=f'-2 STD: {mean_val - 2 * std_val:.2f}')
    ax[1].axvline(mean_val + 2 * std_val, color='orange', linestyle='dashed', linewidth=1,
                  label=f'+2 STD: {mean_val + 2 * std_val:.2f}')

    # Add a legend
    ax[1].legend()

    # Show the plot
    plt.show()
    

def circles_as_type_int(circles):
    circles_int = [[int(x), int(y), int(r)] for x,y,r in circles[0]]
    return circles_int
    

def detect_circles(image_path, path_save=None, debug=None):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape

    # Apply a blur to the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=80)

    # Check if any circles are detected
    if circles is not None and path_save is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            if debug == 1:
                image_gray = gray.copy()
                plot_image_and_histogram(image_gray, x, y, r)

            # Draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # Save the output image
            cv2.imwrite(os.path.join(path_save, 'unfiltered_circles.png'), image)
            cv2.imwrite(f'{path_save[:-4]}_gray.jpg', gray)
            cv2.imwrite(f'{path_save[:-4]}_gray_blurred.jpg', blurred)

    circles_int = circles_as_type_int(circles)
    
    return circles_int, gray, img_shape


def create_bbox_mask(image_shape, preds, conf_thr=0, path_save=None):
    mask = np.zeros(image_shape, dtype=np.uint8)
    height, width = image_shape

    for prediction in preds:
        if prediction['confidence'] < conf_thr:
            continue
        
        bbox = prediction['bbox']
        upper_left = (int(bbox[0][0] * width), int(bbox[0][1] * height))
        bottom_right = (int(bbox[1][0] * width), int(bbox[1][1] * height))
        cv2.rectangle(mask, upper_left, bottom_right, 1, thickness=cv2.FILLED)

    if path_save[0] is not None:
        os.makedirs(path_save[0], exist_ok=True)
        
        bbox_image = cv2.imread(path_save[1])
        for prediction in preds:
            if prediction['confidence'] < conf_thr:
                continue
        
            bbox = prediction['bbox']
            upper_left = (int(bbox[0][0] * width), int(bbox[0][1] * height))
            bottom_right = (int(bbox[1][0] * width), int(bbox[1][1] * height))
            cv2.rectangle(bbox_image, upper_left, bottom_right, (0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(path_save[0], 'bboxes.jpg'), bbox_image,
                   [cv2.IMWRITE_JPEG_QUALITY, 80])

    return mask


def is_circle_inside_mask(mask, circle, min_area_ratio=0.5):
    x, y, r = circle
    height, width = mask.shape
    
    # Create a mask for the circle
    circle_mask = np.zeros_like(mask)
    cv2.circle(circle_mask, (x, y), r, 1, thickness=cv2.FILLED)
    
    # Calculate the area of the circle
    circle_area = np.pi * r * r

    # Calculate the intersection of the circle and the bbox mask
    intersection = np.sum(np.logical_and(mask, circle_mask))

    # Check if at least 50% of the circle's area is inside the bbox
    return intersection >= min_area_ratio * circle_area


def draw_circles_on_image(path_save, circles, save_name):
    image = cv2.imread(path_save[1])
    image_shape = image.shape
    
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 0, 255), thickness=2)
    
    cv2.imwrite(os.path.join(path_save[0], save_name), image, 
               [cv2.IMWRITE_JPEG_QUALITY, 80])

    
def filter_circles(preds, circles, image_shape, path_save=None):
    mask = create_bbox_mask(image_shape, preds, path_save=path_save)
    filtered_circles = [circle for circle in circles if is_circle_inside_mask(mask, circle)]

    if path_save[0] is not None:
        draw_circles_on_image(path_save, circles, 'unfiltered_circles.jpg')
        draw_circles_on_image(path_save, filtered_circles, 'filtered_circles_by_bbox.jpg')
        
    return filtered_circles


def calculate_average_pixel_value(image, circle):
    x, y, r = circle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    total_pixels = np.sum(mask == 255)
    
    if total_pixels == 0:
        return 0
    
    average_pixel_value = np.sum(masked_image) / total_pixels
    
    return average_pixel_value


def filter_circles_by_pixel_value(image, circles, path_save=None):
    circles_with_avg = []
    averages = []
    for circle in circles:
        avg_pixel_value = calculate_average_pixel_value(image, circle)
        circles_with_avg.append((circle, avg_pixel_value))
        averages.append(avg_pixel_value)

    avg_mean = np.mean(averages)
    avg_std = np.std(averages)
    min_val = avg_mean - 2 * avg_std
    max_val = avg_mean + 2 * avg_std

    filtered_circles = [circle for circle, avg in circles_with_avg 
                        if min_val <= avg <= max_val]
    
    if path_save[0] is not None:
        draw_circles_on_image(path_save, filtered_circles, 'filtered_circles_by_pixel_value.jpg')
    
    return filtered_circles


def get_kp_descriptors(img, patchSize=300, edgeThreshold=300, nfeatures=15000):
    if isinstance(img, str):
        img = cv2.imread(img)

    orb = cv2.ORB_create(patchSize=patchSize, edgeThreshold=edgeThreshold, nfeatures=nfeatures)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    return keypoints, descriptors
    

def get_dispenses_params(model, path_folder, img, path_save=None, use_homography=None):
    path_image = os.path.join(path_folder, img)

    if path_save is not None:
        path_save = os.path.join(path_save, img[:-3])
    
    tic = time()
    dispenses_pred = model.predict(path_image).to_dict()
    print(f'Prediction took {time() - tic:.3f} seconds')
    
    circles_in_image, grayscale_image, img_shape = detect_circles(path_image)
    circles_filtered_mask = filter_circles(dispenses_pred['detections'], circles_in_image, img_shape, 
                                      path_save=[path_save, path_image])
    circles_filtered_pixel_value = filter_circles_by_pixel_value(grayscale_image, circles_filtered_mask,
                                                                path_save=[path_save, path_image])

    if use_homography:
        kp, desc = get_kp_descriptors(path_image)
    else:
        kp = []
        desc = []

    return {'Circles': circles_filtered_pixel_value,
            'Num circles': len(circles_filtered_pixel_value),
            'Keypoints': kp,
            'Descriptors': desc}


# In[5]:


''' Functions for matching dispense between reference image and another one '''

def get_homography_matrix(kp_ref, desc_ref, kp_new, desc_new, max_matches=50):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_ref, desc_new)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top matches
    if len(matches) < max_matches:
        max_matches = len(matches)

    good_matches = matches[:max_matches]
    # print(f'Lowest distance - {matches[0].distance}')

    # Extract matched points
    points_golden = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_new = np.float32([kp_new[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    homography_new2ref, _ = cv2.estimateAffinePartial2D(points_new, points_golden)

    return homography_new2ref


def apply_homography(points, homography_matrix):
    # Convert points to homogeneous coordinates
    n_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((n_points, 1))])

    # Apply the homography matrix
    transformed_homogeneous_points = homogeneous_points @ homography_matrix.T

    # Convert back to 2D coordinates
    transformed_points = transformed_homogeneous_points[:, :2]

    return transformed_points


def find_closest_pairs(points1, points2, max_dist=80):
    # max_dist was created heuristically so a dispense won't be matched to its neighbour

    matches = []
    used_indices = set()

    # calculate distance matrix
    dist_matrix = np.sqrt(np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :]) ** 2, axis=2))

    for i in range(len(points1)):
        min_dist = max_dist
        best_match = None
        for j in range(len(points2)):
            if j not in used_indices and dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                best_match = j
        if best_match is not None:
            used_indices.add(best_match)
            matches.append((best_match, i))

    return matches

    
def calc_deviations(circles_ref, circles_new, homography_mat=None):
    deviations = np.inf * np.ones((len(circles_new),3))

    circles_ref_xy = np.array([[x, y] for x, y, _ in circles_ref])
    circles_new_xy = np.array([[x, y] for x, y, _ in circles_new])

    if homography_mat is not None:
        circles_transposed = apply_homography(circles_new_xy, homography_mat)
    else:
        circles_transposed = circles_new_xy

    matched_points = find_closest_pairs(circles_transposed, circles_ref_xy)

    for ind_ref_matched, ind_new in matched_points:
        deviations[ind_new, :2] = circles_transposed[ind_new, :] - circles_ref_xy[ind_ref_matched, :]
        deviations[ind_new, 2] = circles_new[ind_new][2]

    return deviations, matched_points


def draw_matches(path_img1, points1, path_img2, points2, matches, path_save_folder):
    os.makedirs(path_save_folder, exist_ok=True)
    path_save = os.path.join(path_save_folder, f'{os.path.basename(path_img2)} - matches.jpg')

    img1 = cv2.imread(path_img1)
    img2 = cv2.imread(path_img2)

    # Create a new image by concatenating img1 and img2 horizontally
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    combined_img = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    combined_img[:height1, :width1, :] = img1
    combined_img[:height2, width1:width1 + width2, :] = img2

    # Draw matching points with green lines
    for (idx1, idx2) in matches:
        pt1 = tuple(points1[idx1])
        pt2 = tuple(points2[idx2] + np.array([width1, 0]))  # Offset the second point's x-coordinate
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)

    # Draw unmatched points with red circles
    matched_points1 = {idx1 for idx1, _ in matches}
    matched_points2 = {idx2 for _, idx2 in matches}

    for i, pt in enumerate(points1):
        if i not in matched_points1:
            cv2.circle(combined_img, tuple(pt), 5, (0, 0, 255), 1)

    for i, pt in enumerate(points2):
        if i not in matched_points2:
            cv2.circle(combined_img, tuple(pt + np.array([width1, 0])), 5, (0, 0, 255), 1)

    cv2.imwrite(path_save, combined_img)
    

def get_deviations(circles_per_image, ref_name, save_matches, use_homography=False):
    
    ref_circles = circles_per_image[ref_name]['Circles']
    ref_kp = circles_per_image[ref_name]['Keypoints']
    ref_desc = circles_per_image[ref_name]['Descriptors']

    ref_dev, _ = calc_deviations(ref_circles, ref_circles)
    deviations_dict = [{'image name': ref_name, 'deviations': ref_dev}]

    for id_circ, (img_name, circles_params) in enumerate(circles_per_image.items()):
        if img_name == ref_name:
            continue
        else:
            curr_circles = circles_per_image[img_name]['Circles']
            curr_kp = circles_per_image[img_name]['Keypoints']
            curr_desc = circles_per_image[img_name]['Descriptors']

            if use_homography:
                homography_mat = get_homography_matrix(ref_kp, ref_desc, curr_kp, curr_desc)
            else:
                homography_mat = None

            deviations, matched_points = calc_deviations(ref_circles, curr_circles,
                                                         homography_mat=homography_mat)

            deviations_dict.append({'image name': img_name, 'deviations': deviations})

            if save_matches[0] is not None:
                path_src = save_matches[1]
                path_save = os.path.join(save_matches[0], 'Matched dispenses')
                path_img_ref = os.path.join(path_src, ref_name)
                path_img_new = os.path.join(path_src, img_name)
                points_ref = np.array([[x, y] for x, y, _ in ref_circles])
                points_new = np.array([[x, y] for x, y, _ in curr_circles])
    
                draw_matches(path_img_ref, points_ref, path_img_new, points_new, matched_points, path_save)

    return deviations_dict


# In[6]:


def file_is_image(file_path, acceptable_formats=['.jpg','.jpeg','.bmp','.png']):
    _, file_extension = os.path.splitext(file_path)

    return file_extension in acceptable_formats


def create_dispense_col_names(num_dispenses):
    col_names = []

    for i in range(num_dispenses):
        names_per_i = [f'Dot{i+1}_dx', f'Dot{i+1}_dy', f'Dot{i+1}_r']
        col_names.extend(names_per_i)

    return col_names


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def save_results_to_excel(list_to_save, file_path, num_dispenses):
    data = []
    dispenses_col_names = create_dispense_col_names(num_dispenses)
    col_names = ['Image name'] + dispenses_col_names
    
    for curr_dict in list_to_save:
        img_name = curr_dict['image name']
        dispenses_params = curr_dict['deviations']
        dispenses_params_list = flatten_list(dispenses_params)
        data.append([img_name] + dispenses_params_list)

    df = pd.DataFrame(data, columns=col_names)
    df.to_excel(file_path, index=False)


def run_folder_analysis(path_folder, path_save_excel,
                        num_expected_dispenses=44,
                        deployment_server_url='https://ses1-ate.solaredge.com/api/mlflow/relay', 
                        model_name='Dispense detection', 
                        model_stage="Staging",
                        save_mode=None,
                        refernce_img=None,
                        path_save=None,
                        use_homography=False):

    model = download_model_from_url(deployment_server_url, model_name, model_stage)
    imgs = os.listdir(path_folder)
    results_circles = dict()

    for img in imgs:
        if not file_is_image(img):
            print(f'File is not an image: {img}')
            continue
        
        results_circles[img] = get_dispenses_params(model, path_folder, img,
                                                    use_homography=use_homography,
                                                    path_save=path_save)
        
        if reference_img is None and results_circles[img]['Num circles'] == num_expected_dispenses:
            reference_img = img

    if reference_img is None:
        print('No image had enough dispenses')
    else:
        results_deviations = get_deviations(results_circles, reference_img,
                                            save_matches=[path_save, path_folder],
                                            use_homography=use_homography)
        save_results_to_excel(results_deviations, f'{path_save_excel}.xlsx', num_expected_dispenses)