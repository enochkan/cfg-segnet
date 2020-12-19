import os, cv2, numpy as np
path = './images/'
files = ['cfg.png', 'au.png', 'unet.png']

for file in files:
    img = cv2.imread(path + file)
    if file != 'cfg.png':
        # print(img.shape)

        ct = img[1:257,2:258,:]
        gt_original = img[257:513,2:258,:]
        proposed_original = img[515:771, 2:258, :]


        gt = cv2.cvtColor(gt_original, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gt, 50, 255, cv2.THRESH_BINARY)
        contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        p = cv2.cvtColor(proposed_original, cv2.COLOR_BGR2GRAY)
        _, thresh_p = cv2.threshold(p, 50, 255, cv2.THRESH_BINARY)
        contours_proposed,_ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw contours:
        added_original = gt_original + proposed_original
        added_original = cv2.drawContours(np.array(added_original), contours, -1, (0, 0, 255), 1)
        added_original = cv2.drawContours(np.array(added_original), contours_proposed, -1, (255, 0, 0), 1)
        added_original = cv2.bitwise_or(ct, added_original)
        # added_image = cv2.addWeighted(gt,.85,p,1,0)

        window_name = file.split('.')[0] + 'masked.png'

        # Using cv2.imshow() method
        # Displaying the image
        cv2.imwrite(window_name, added_original)
    else:
        print(img.shape)
        ct = img[1:257,2:258,:]
        # gt_original = img[257:513,2:258,:]
        gt_original = img[516:772, 2:258, :]
        proposed_original = img[772:1028, 2:258, :]
        print(proposed_original.shape)
        # print(img.shape)

        gt = cv2.cvtColor(gt_original, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gt, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        p = cv2.cvtColor(proposed_original, cv2.COLOR_BGR2GRAY)
        _, thresh_p = cv2.threshold(p, 50, 255, cv2.THRESH_BINARY)
        contours_proposed, _ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw contours:
        added_original = gt_original + proposed_original
        added_original = cv2.drawContours(np.array(added_original), contours, -1, (0, 0, 255), 1)
        added_original = cv2.drawContours(np.array(added_original), contours_proposed, -1, (255, 0, 0), 1)
        added_original = cv2.bitwise_or(ct, added_original)
        # added_image = cv2.addWeighted(gt,.85,p,1,0)

        window_name = file.split('.')[0] + 'masked.png'

        # Using cv2.imshow() method
        # Displaying the image
        cv2.imwrite(window_name, added_original)

