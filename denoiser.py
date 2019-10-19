from heapq import heappushpop, heappush
from itertools import count

from scipy.special import logsumexp

from util import *

_tiebreaker = count()  # Tiebreaker for heaps


class Denoiser:
    """
    Implements NL-means algorithm as specified in section 3 of
    A Non-Local Algorithm For Image Denoising (Baudes et al., 2005)
    by using Generalized PatchMatch (as specified in section 3.2 of The Generalized PatchMatch Correspondence Algorithm
    (Barnes et al., 2010)) with target = source = noisy image
    """

    def __init__(self, image: str, iters: int = 3, alpha: float = 0.5, w: int = None, patch_size: int = 7,
                 k: int = 4, h: float = 0.5, propagation_enabled: bool = True, random_enabled: bool = True):
        """
        :param image: path to source image
        :param iters: number of iterations of patchmatch to run
        :param alpha: ratio between patch sizes (see Equation 1 in patchmatch paper)
        :param w: max search radius (set to None to use max dimension of image) (see Equation 1 in patchmatch paper)
        :param patch_size: patch size
        :param k: how many nearest neighbours to use
        :param h: weight scaling factor as described in the equation for w(i, j) in section 3 of NLM paper
        :param propagation_enabled: whether to perform the propagation step (for debugging)
        :param random_enabled: whether to perform the random search step (for debugging)
        """
        self.image = load_image(image)
        self.iters = iters
        self.alpha = alpha
        self.w = np.max(self.image.shape[0:2]) if w is None else w
        self.patch_size = patch_size
        self.k = k
        self.h = h
        self.propagation_enabled = propagation_enabled
        self.random_enabled = random_enabled
        self.nnf = None
        self.nnf_heap = None
        self.nnf_coord_dicts = None
        self.radii = None
        self.image_patches = None

    def init_radii(self) -> None:
        """
        Set self.radii to w * alpha**i for i=0,1,... until radius < 1 pixel
        """
        search_radius = self.w
        radii = []
        while search_radius >= 1:
            radii.append(search_radius)
            search_radius *= self.alpha
        radii = np.asarray(radii, dtype=np.uint16)
        self.radii = np.repeat(radii[:, np.newaxis], 2, axis=1)

    def denoise(self) -> np.ndarray:
        """
        Denoise the image at self.source by first executing patchmatch then running nlm
        :return: Denoised image as ndarray
        """
        print("Initializing...", end=' ')
        self.init_radii()  # Initialize radii
        self.image_patches = generate_patches(self.image, self.patch_size)  # Initialize patches
        self.nnf = self.random_nnf()
        self.init_heap(self.k_nnf())
        assert self.nnf is not None and self.image is not None
        # Run the algorithm
        print("done!\nRunning patchmatch...")
        for i in range(self.iters):
            self.run_patchmatch_iteration(i % 2 != 0)
            print("Finished iteration {}/{}".format(i + 1, self.iters))
        print("Done PatchMatch!\nDenoising...", end=' ')
        result = self.nlm()
        print("done!")
        return result

    def run_patchmatch_iteration(self, odd_iteration: bool = False) -> None:
        """
        Run patchmatch loop iteration as specified in section 3.2 of the patchmatch paper
        :param odd_iteration: whether we are on an odd iteration
        :return: (new nearest neighbour field, new similarity matrix, new global_vars)
        """
        y_len = len(self.nnf_heap)
        x_len = len(self.nnf_heap[0])
        coords = coords_matrix([y_len, x_len])
        # Scan order for odd iterations
        if odd_iteration:
            y_range = np.arange(y_len)
            x_range = np.arange(x_len)
        # Reverse scan order for even iterations
        else:
            y_range = np.arange(start=y_len - 1, stop=-1, step=-1)
            x_range = np.arange(start=x_len - 1, stop=-1, step=-1)
        k_range = range(len(self.nnf_heap[0][0]))
        # Iterate over patch centres
        for y in y_range:
            for x in x_range:
                '''
                #############
                 PROPAGATION
                #############
                '''
                if self.propagation_enabled:
                    for k in k_range:
                        candidates = []
                        # Left and up for odd
                        if odd_iteration:
                            if x > 0:
                                candidates.append(coords[y, x] + self.nnf_heap[y][x - 1][k][2])
                            if y > 0:
                                candidates.append(coords[y, x] + self.nnf_heap[y - 1][x][k][2])
                        # Right and down for even
                        else:
                            if x < x_len - 2:
                                candidates.append(coords[y, x] + self.nnf_heap[y][x + 1][k][2])
                            if y < y_len - 2:
                                candidates.append(coords[y, x] + self.nnf_heap[y + 1][x][k][2])
                        # If there are no candidates then there will still be no candidates for other k values so break
                        if len(candidates) == 0:
                            break
                        candidates = np.asarray(candidates, dtype=int)
                        self.process_nnf_candidates(candidates, y, x)

                '''
                ###############
                 RANDOM SEARCH
                ###############
                '''
                if self.random_enabled:
                    # Get search radii
                    for k in k_range:
                        candidates = self.nnf_heap[y][x][k][2] + self.radii * np.random.uniform(-1, 1, (
                            self.radii.shape[0], 2))  # u_i
                        np.round(candidates, out=candidates)
                        self.process_nnf_candidates(candidates.astype(int), y, x)

    def process_nnf_candidates(self, candidates: np.ndarray, y: int, x: int) -> None:
        """
        Process candidates and update the nnf for (y, x) to be the min of its value and candidates
        :param candidates: ndarray of ints of candidate points
        :param y: y index in the image
        :param x: x index in the image
        """
        # Ensure candidates are in bounds
        np.clip(candidates, [-y, -x], [len(self.nnf_heap) - y - 1, len(self.nnf_heap[0]) - x - 1], out=candidates)
        # Get coordinates of candidate offsets
        candidates += [y, x]
        diff = np.abs(
            self.image_patches[candidates[..., 0], candidates[..., 1]] - self.image_patches[y, x])
        diff[np.isnan(diff)] = 255
        distances = -np.linalg.norm(diff, axis=(1, 2))
        min_dist_index = np.argmax(distances)
        min_dist = distances[min_dist_index]
        off = candidates[min_dist_index] - [y, x]
        if tuple(off) not in self.nnf_coord_dicts[y][x] and min_dist > self.nnf_heap[y][x][0][0]:
            t = tuple(self.nnf_heap[y][x][0][2])
            self.nnf_coord_dicts[y][x][t] -= 1
            if self.nnf_coord_dicts[y][x][t] <= 0:
                del self.nnf_coord_dicts[y][x][t]
            heappushpop(self.nnf_heap[y][x], (min_dist, self.nnf_heap[y][x][0][1], off))
            self.nnf_coord_dicts[y][x][
                tuple(off)] = 1  # Already checked that off isn't an existing key

    def random_nnf(self) -> np.ndarray:
        """
        Generate random coordinates for each pixel to create a random initial nnf
        """
        # Stack random x and y coordinates
        coords = np.dstack((np.random.randint(low=0, high=self.image.shape[0],
                                              size=(self.image.shape[0], self.image.shape[1])),
                            np.random.randint(low=0, high=self.image.shape[1],
                                              size=(self.image.shape[0], self.image.shape[1]))))

        return coords - coords_matrix(self.image.shape)

    def k_nnf(self) -> np.ndarray:
        """
        Returns k*self.image.shape ndarray n where n[i] is a self.image.shape nnf
        :return: k*self.image.shape ndarray n where n[i] is a self.image.shape nnf
        """
        # Get 1 random nnf first so we know dimensions and datatype
        nnf = self.random_nnf()
        nnf_k = np.empty((self.k, nnf.shape[0], nnf.shape[1], nnf.shape[2]), nnf.dtype)
        nnf_k[0] = nnf
        # Generate k-1 more random nnfs of the same shape/dtype
        for i in range(1, self.k):
            nnf_k[i] = self.random_nnf()
        return nnf_k

    def init_heap(self, nnf_k: np.ndarray) -> None:
        """
        Initialize the nnf heap to the correct values for nnf nnf_k
        :param nnf_k: nnf to use to initialize heap values
        """

        def dist(source_y, source_x, target_y_off, target_x_off):
            d = np.abs(
                self.image_patches[target_y_off + source_y, target_x_off + source_x] - self.image_patches[
                    source_y, source_x])
            d[np.isnan(d)] = 255  # 8 bit image => 255 is max possible dist
            result = np.linalg.norm(d)
            return result

        heap = np.empty((nnf_k.shape[1], nnf_k.shape[2])).tolist()
        coords = np.empty((nnf_k.shape[1], nnf_k.shape[2])).tolist()
        f = np.moveaxis(nnf_k, 0, 2)
        for y in range(f.shape[0]):
            for x in range(f.shape[1]):
                coords[y][x] = {}
                heap[y][x] = []
                for k in range(f.shape[2]):
                    heappush(heap[y][x], (-dist(y, x, f[y, x, k, 0], f[y, x, k, 1]), k, f[y, x, k]))
                    t = tuple(f[y, x, k])
                    if t not in coords[y][x]:
                        coords[y][x][t] = 1
                    else:
                        coords[y][x][t] += 1
        self.nnf_coord_dicts = coords
        self.nnf_heap = heap

    def nlm(self) -> np.ndarray:
        """
        Returns image processed with the NL-means algorithm using nnf_heap
        as specified in part 3 of the NLM paper
        :return: image processed with NL-means algorithm using nnf_heap
        """
        assert self.image is not None and self.nnf_heap is not None
        coords = coords_matrix([len(self.nnf_heap), len(self.nnf_heap[0])])
        denoised = np.empty_like(self.image)
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                # To avoid floating point problems work in log space then exp to get weights
                logs = [
                    self.nnf_heap[y][x][k][0] / self.h ** 2  # Priority = -1 * norm of difference
                    for k in range(len(self.nnf_heap[y][x]))
                ]
                total = logsumexp(logs)
                weights = np.repeat(np.reshape([np.exp(l - total) for l in logs], (len(logs), 1)), 3,
                                    axis=1)
                vals = np.array(
                    [self.image[pixel[0], pixel[1]] for pixel in
                     [coords[y][x] + heap[2] for heap in self.nnf_heap[y][x]]])
                denoised[y][x] = np.round(np.sum(weights * vals, axis=0))
        return denoised
