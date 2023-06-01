import os
import sys
from PIL import Image
import numpy as np
import time
from math import sqrt
import cv2
from skimage.transform import radon, rescale
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ResultNotExist(Exception):
    pass


class WrongWindowSize(Exception):
    pass


class WrongRank(Exception):
    pass


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
            # можно и в чат послать
        return result

    return timed


class ImageProcessor:
    def __init__(self, path=None, image=None):
        self.path = path
        self.orig = None
        self.result = None
        self.grayscale_matrix = None
        self.binary_matrix = None
        self.filtered_matrix = None
        self.gradient_matrix = None

        if path is not None:
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(sys.path[0], path))

            self.orig = Image.open(path).convert("RGB")
            self.rgb_matrix = np.array(self.orig)

        if image is not None:
            self.orig = image
            self.rgb_matrix = np.array(self.orig)

    def show(self):
        if self.result is None:
            self.orig.show()
        else:
            self.result.show()

    def save(self, filepath: str):
        if self.result is not None:
            self.result.save(filepath)
        else:
            raise ResultNotExist("No such results for saving it to {}".format(filepath))

    @timeit
    def negate(self):
        w, h = self.orig.size
        pxs = self.orig.load()
        img_ht = Image.new('RGB', (w, h), color='white')
        pxs_ht = img_ht.load()
        for i in range(w):
            for j in range(h):
                pxs_ht[i, j] = (255 - pxs[i, j][0], 255 - pxs[i, j][1], 255 - pxs[i, j][2])

        self.result = img_ht

    @timeit
    def to_grayscale(self):
        self.result = self.orig.convert("L")

    @timeit
    def to_grayscale_np(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        self.grayscale_matrix = gray_matrix

        gray_im = Image.fromarray(np.uint8(gray_matrix), 'L')
        self.result = gray_im

    @timeit
    def to_grayscale_loop(self):
        w, h = self.orig.size
        pxs = self.orig.load()
        img_ht = Image.new('RGB', (w, h), color='white')
        pxs_ht = img_ht.load()
        for i in range(w):
            for j in range(h):
                g = int((pxs[i, j][0] + pxs[i, j][1] + pxs[i, j][2]) / 3)
                pxs_ht[i, j] = (g, g, g)

        self.result = img_ht

    @timeit
    def to_binary(self):
        self.result = self.orig.convert("1")

    @timeit
    def to_binary_loop(self, threshold=128):
        w, h = self.orig.size
        pxs = self.orig.load()
        img_ht = Image.new('RGB', (w, h), color='white')
        pxs_ht = img_ht.load()
        for i in range(w):
            for j in range(h):
                g = int((pxs[i, j][0] + pxs[i, j][1] + pxs[i, j][2]) / 3)
                pxs_ht[i, j] = (0, 0, 0) if g < threshold else (255, 255, 255)

        self.result = img_ht

    @timeit
    def to_binary_otsu(self, rsize=3, Rsize=15, eps=15):
        def otsu_global(matrix: np.ndarray):
            n_curr = 0
            T_res = 0
            M0_res = 0
            M1_res = 0

            p_tmp = np.unique(matrix, return_counts=True)
            p = p_tmp[1] / matrix.size

            for t in range(matrix.min(), matrix.max()):
                w0 = p[p_tmp[0] <= t].sum() if p[p_tmp[0] <= t].sum() > 0.00001 else 0.00001
                w1 = 1 - w0 if 1 - w0 > 0.00001 else 0.00001
                M0 = (p_tmp[0][p_tmp[0] <= t] * p[p_tmp[0] <= t]).sum() / w0
                M1 = (p_tmp[0][p_tmp[0] > t] * p[p_tmp[0] > t]).sum() / w1
                D0 = (p[p_tmp[0] <= t] * np.square(p_tmp[0][p_tmp[0] <= t] - M0)).sum()
                D1 = (p[p_tmp[0] > t] * np.square(p_tmp[0][p_tmp[0] > t] - M1)).sum()

                n = (w0 * w1 * (M0 - M1) ** 2) // (w0 * D0 + w1 * D1)
                if n >= n_curr:
                    n_curr = n
                    T_res = t
                    M0_res = M0
                    M1_res = M1

            return T_res, M0_res, M1_res

        # @timeit
        def split_submatrix(matrix: np.ndarray, submat1_shape: tuple, submat2_shape: tuple):
            p, q = submat1_shape
            P, Q = submat2_shape
            m, n = matrix.shape

            bias_p = (P - p) // 2
            bias_q = (Q - q) // 2
            for x in range(0, m, p):
                for y in range(0, n, q):
                    yield (
                        (
                            (x, (x + p) if (x + p - m) < 0 else m),
                            (y, (y + q) if (y + q - n) < 0 else n)
                        ),
                        (
                            ((x - bias_p) if (x - bias_p) > 0 else 0, (x + P - bias_p) if (x + P - bias_p) < m else m),
                            ((y - bias_q) if (y - bias_q) > 0 else 0, (y + Q - bias_q) if (y + Q - bias_q) < n else n)
                        )
                    )

        def binarization_processor(matrix_ind: tuple, epsilon=eps):
            matrix_k_ind, matrix_K_ind = matrix_ind
            matrix_k = self.grayscale_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1],
                       matrix_k_ind[1][0]: matrix_k_ind[1][1]]
            matrix_K = self.grayscale_matrix[matrix_K_ind[0][0]: matrix_K_ind[0][1],
                       matrix_K_ind[1][0]: matrix_K_ind[1][1]]
            T, M0, M1 = otsu_global(matrix_K)

            if abs(M1 - M0) >= epsilon:
                self.binary_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]] = \
                    np.where(matrix_k < T, 0, 255)
            else:
                k_mean = matrix_k.mean()
                new_T = (M0 + M1) / 2
                self.binary_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]] \
                    .fill(0 if k_mean <= new_T else 255)

        if (not (rsize % 2) and not (Rsize % 2)) or ((rsize % 2) and (Rsize % 2)):
            if self.grayscale_matrix is None:
                self.to_grayscale_np()
            self.binary_matrix = self.grayscale_matrix.astype(np.uint8)

            for x in split_submatrix(self.binary_matrix, (rsize, rsize), (Rsize, Rsize)):
                binarization_processor(x)

            self.result = Image.fromarray(self.binary_matrix, 'L')

        else:
            raise WrongWindowSize("Rsize={} and rsize={} must be even or odd both together".format(Rsize, rsize))

    @timeit
    def rank_filter(self, rank, wsize=3):
        def prepare_matrix(matrix: np.ndarray, window_size: int):
            bias = wsize // 2
            new_matrix = np.vstack((matrix[1:(bias + 1)][::-1], matrix, matrix[-(bias + 1):-1][::-1]))
            new_matrix = np.hstack(
                (new_matrix[:, 1:(bias + 1)][:, ::-1], new_matrix, new_matrix[:, -(bias + 1):-1][:, ::-1]))

            return new_matrix

        if not wsize % 2:
            raise WrongWindowSize("wsize must be odd, positive and integer")

        if rank >= wsize ** 2 or rank < 0:
            raise WrongRank("rank must be positive and less than wsize*wsize")

        if self.grayscale_matrix is None:
            self.to_grayscale_np()

        bias = wsize // 2
        prepared_matrix = prepare_matrix(self.grayscale_matrix, wsize)
        filtered_matrix = np.ndarray(self.grayscale_matrix.shape)
        for (x, y), _ in np.ndenumerate(self.grayscale_matrix):
            filtered_matrix[x, y] = sorted(prepared_matrix[x: x + wsize, y: y + wsize].flatten())[rank]

        self.filtered_matrix = np.uint8(filtered_matrix)
        self.result = Image.fromarray(self.filtered_matrix, 'L')

    @timeit
    def scharr_operator(self, threshold=None):
        gx = np.array([[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]])
        gy = np.array([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]])

        if self.grayscale_matrix is None:
            self.to_grayscale_np()

        gradient_matrix = np.empty(self.grayscale_matrix.shape)
        for (x, y), _ in np.ndenumerate(self.grayscale_matrix[1: -1, 1: -1]):
            a = self.grayscale_matrix[x: x + 3, y: y + 3]
            gradient_matrix[x + 1, y + 1] = np.sqrt(np.sum(gx * a) ** 2 + np.sum(gy * a) ** 2)
        self.gradient_matrix = gradient_matrix * 255 / np.max(gradient_matrix)

        if threshold is None:
            self.result = Image.fromarray(np.uint8(self.gradient_matrix), 'L')
        else:
            gradient_matrix = np.where(self.gradient_matrix < threshold, 0, 255)
            self.result = Image.fromarray(np.uint8(gradient_matrix), 'L')

    @timeit
    def grad_operator(self, threshold=50):
        w, h = self.orig.size
        pxs = self.orig.load()
        img_pr = Image.new('RGB', (w, h), color='white')
        pxs_pr = img_pr.load()
        img_pr_T = Image.new('RGB', (w, h), color='white')
        pxs_pr_T = img_pr_T.load()
        max_intensity = 0

        for i in range(w):
            for j in range(h):
                border = True if i == 0 or j == 0 or i == w - 1 or j == h - 1 else False
                if not border:
                    Gx = pxs[i + 1, j - 1][0] + pxs[i + 1, j][0] + pxs[i + 1, j + 1][0] - pxs[i - 1, j - 1][0] - \
                         pxs[i - 1, j][0] - pxs[i - 1, j + 1][0]
                    Gy = pxs[i - 1, j + 1][0] + pxs[i, j + 1][0] + pxs[i + 1, j + 1][0] - pxs[i - 1, j - 1][0] - \
                         pxs[i, j - 1][0] - pxs[i + 1, j - 1][0]
                    Gr = int(sqrt(Gx ** 2 + Gy ** 2))
                else:
                    Gr = pxs[i, j][0]
                max_intensity = Gr if max_intensity < Gr else max_intensity
                pxs_pr[i, j] = (Gr, Gr, Gr)

        for i in range(w):
            for j in range(h):
                Gr_n = int(pxs_pr[i, j][0] * 255 / max_intensity)
                color = 0 if Gr_n < threshold else 255
                pxs_pr[i, j] = (Gr_n, Gr_n, Gr_n)
                pxs_pr_T[i, j] = (color, color, color)

    @timeit
    def prewitt_operator(self):
        """ calculate gradient matrix with Prewitt operator """
        pix = self.orig.load()
        w, h = self.orig.size
        gx = np.zeros((h, w))
        gy = np.zeros((h, w))

        for x in range(1, w - 1):
            for y in range(1, h - 1):
                gx[y, x] = pix[x - 1, y - 1] + pix[x, y - 1] + pix[x + 1, y - 1] - \
                           (pix[x - 1, y + 1] + pix[x, y + 1] + pix[x + 1, y + 1])
                gy[y, x] = pix[x - 1, y - 1] + pix[x - 1, y] + pix[x - 1, y + 1] - \
                           (pix[x + 1, y - 1] + pix[x + 1, y] + pix[x + 1, y + 1])

        g = np.sqrt(np.square(gx) + np.square(gy))
        g_min, g_max = g.min(), g.max()
        threshold = g_min + (g_max - g_min) // 12
        g = (g >= threshold) * 255
        p_img = Image.fromarray(np.asarray(g, dtype="uint8"), "L")
        self.result = p_img
        return g

    @timeit
    def calc_min_max(self):
        try:
            data = np.asarray(self.orig, dtype='uint8')
        except SystemError:
            data = np.asarray(self.orig.getdata(), dtype='uint8')
        return data.min(), data.max()

    @timeit
    def calc_histogram(self):
        """ map RGB cube to 4*4*4=64 array by dividing colors by 64 (>>6) """
        w, h = self.orig.size
        pix = self.orig.load()
        hist = np.zeros((64,), dtype=int)
        for x in range(w):
            for y in range(h):
                r, g, b = pix[x, y]
                hist[(r >> 6 << 4) + (g >> 6 << 2) + (b >> 6)] += 1
        # normalize and quantize by 256 values
        hist = (hist << 8) // (w * h)
        return hist

    def calc_GLCM(self):
        """ calculate Gray-Level Co-Occurrence Matrix (Haralick) """
        np_glcm = np.zeros((256, 256))
        w, h = self.orig.size
        pix = self.orig.load()
        for x in range(w - 1):
            for y in range(h - 1):
                np_glcm[pix[x, y]][pix[x + 1, y]] += 1
                np_glcm[pix[x, y]][pix[x, y + 1]] += 1

        glcm_img = Image.fromarray(np.asarray(np.clip(np_glcm, 0, 255), dtype="uint8"), "L")
        total = (w - 1) * (h - 1) * 2
        np_glcm = np_glcm / total
        return np_glcm, glcm_img

    def calc_texture_hash(self, glcm):
        # Энергия: SS Pij^2, [0; 1)
        asm = np.sum(glcm ** 2)

        # Контрастность: SS Pij(i-j)^2
        con = 0.0
        for i in range(256):
            for j in range(256):
                con += glcm[i][j] * (i - j) ** 2

        # Максимальная вероятность [0; 1)
        mpr = np.max(glcm)

        # Локальная однородность (гомогенность): SS P/(1+(i-j)^2), [0; 1)
        lun = np.sum(glcm / [[1 + (i - j) ** 2 for j in range(256)] for i in range(256)])

        # Энтропия: -S P log2 P
        ent = - np.sum(glcm * np.vectorize(lambda x: np.log2(x) if x > 0 else 0)(glcm))

        # След матрицы: S Pii, [0; 1)
        tr = 0.0
        for i in range(256):
            tr += glcm[i][i]

        # Мат.ожидание серого
        mux = np.sum(glcm, axis=1)
        muy = np.sum(glcm, axis=0)
        mu1 = np.sum(np.multiply(mux, [i for i in range(256)]))
        mu2 = np.sum(np.multiply(muy, [i for i in range(256)]))

        # СКО
        variance = 0.0
        for i in range(256):
            variance += mux[i] * (i - mu1) ** 2
        std = sqrt(variance)

        # Корреляция значений яркости:
        corr = 0.0
        for i in range(256):
            for j in range(256):
                corr += glcm[i][j] * (i - mu1) * (j - mu1)
        if variance > 0:
            corr /= variance
        else:
            corr = 1

        # quantifying
        asm = int(asm * 65536)
        con = int(con)
        mpr = int(mpr * 65536)
        ent = int(ent * 256)
        lun = int(lun * 256)
        tr = int(tr * 256)
        mu1 = int(mu1)
        mu2 = int(mu2)
        std = int(std)
        corr = int(corr * 256)

        # return [asm * 1000.0, ent / 10.0, lun, tr]
        return [asm, con, mpr, lun, ent, tr, mu1, mu2, std, corr]


@timeit
def gabor_filter(filepath, res_path):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    cv2.imwrite(res_path, filtered_img)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(filtered_img, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(res_path, g_kernel)


@timeit
def sinogram(filepath, res_path):
    image = np.array(Image.open(filepath).convert("L"))
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True, preserve_range=True)
    plt.imshow(sinogram, cmap=plt.get_cmap('magma'), extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    plt.savefig(res_path, facecolor='w')


if __name__ == '__main__':
    pass
