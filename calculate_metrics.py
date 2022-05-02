import argparse
import sys
import datetime

import imageio
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score
import scipy.stats as stats
from skimage.morphology import dilation, closing, square, reconstruction, binary_dilation, disk, binary_closing
import cv2

# from utils import *

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def evaluate(lbl, pred, method=''):
    acc = accuracy_score(lbl, pred)
    conf_m = confusion_matrix(lbl, pred)
    f1_s_w = f1_score(lbl, pred, average='weighted')
    f1_s_micro = f1_score(lbl, pred, average='micro')
    f1_s_macro = f1_score(lbl, pred, average='macro')
    kappa = cohen_kappa_score(lbl, pred)
    jaccard = jaccard_score(lbl, pred)
    tau, p = stats.kendalltau(lbl, pred)

    _sum = 0.0
    for k in range(len(conf_m)):
        _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

    print("---- Validation/Test -- Method " + method +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(_sum / 2.0) +
          " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
          " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
          " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
          " Kappa= " + "{:.4f}".format(kappa) +
          " Jaccard= " + "{:.4f}".format(jaccard) +
          " Tau= " + "{:.4f}".format(tau) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--gt', type=str, required=True, help='Ground Truth.')
    parser.add_argument('--pred', type=str, required=True, help='Prediction map.')
    args = parser.parse_args()
    print(sys.argv[0], args)

    lbl = imageio.imread(args.gt).astype(int)[:, :, 0]
    pred = imageio.imread(args.pred).astype('uint8')

    lbl[np.where(lbl == 255)] = 1
    pred[np.where(pred == 255)] = 1

    print('lbl', lbl.shape, np.bincount(lbl.flatten()))
    print('pred', pred.shape, np.bincount(pred.flatten()))

    clo_100_1 = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8), iterations=1)
    clo_200_1 = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((200, 200), np.uint8), iterations=1)

    clo_100_3 = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((100, 100), np.uint8), iterations=3)
    # print('clo_100', clo_100.shape, np.bincount(clo_100.flatten()))
    clo_200_3 = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones((200, 200), np.uint8), iterations=3)
    # print('clo_200', clo_200.shape, np.bincount(clo_200.flatten()))
    # imageio.imwrite('/home/kno/datasets/rios/largos_estreitos/se_clo_square_200.png', clo*255)

    dil_100_3 = cv2.dilate(pred, np.ones((100, 100), np.uint8), iterations=3)
    # imageio.imwrite('/home/kno/datasets/rios/largos_estreitos/se_dil_100_3.png', dil_100_3*255)
    rec = reconstruction(dil_100_3, pred, method='erosion')
    # imageio.imwrite('/home/kno/datasets/rios/largos_estreitos/se_rec.png', rec)
    # print('rec', rec.shape, np.bincount(rec.flatten().astype(int)))

    evaluate(lbl.flatten(), pred.flatten(), 'Original')
    evaluate(lbl.flatten(), clo_100_1.flatten(), 'Closing_100_1')
    evaluate(lbl.flatten(), clo_200_1.flatten(), 'Closing_200_1')
    evaluate(lbl.flatten(), clo_100_3.flatten(), 'Closing_100_3')
    evaluate(lbl.flatten(), clo_200_3.flatten(), 'Closing_200_3')
    evaluate(lbl.flatten(), rec.flatten(), 'Rec')
