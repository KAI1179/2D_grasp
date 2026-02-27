import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
import cv2


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True, center=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None
        if center:
            center_out = np.zeros(shape)
        else:
            center_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            rr1, cc1 = gr.center#####################
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length
            if center:
                center_out[rr1,cc1] = 1

        return pos_out, ang_out, width_out, center_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def refine_polygon_coords(a, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(a.points[:, 0], a.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(np.float))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr #网络预测生成的
        max_iou = 0
        for gr in grs: #gr/grs是gtbbs
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)

################################################################################
def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):  # 生成初框
    global g
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    grasp = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_img[grasp_point]
        g = Grasp(grasp_point, grasp_angle)

        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2
    grasp.append(g)
    return grasp  # 返回列表

def find_region_center(region):  # 输入区域，输出区域中心点
    global cY, cX
    M = cv2.moments(region)
    if M["m00"] == 0:
        print('first grasp box is empty and there is no region')
        cX = 150
        cY = 150
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cY, cX

def generate_grasps(ang_img, region_img, width_img=None):  # 输入区域，返回以区域中心为中心点的抓取框
    grasp_center = find_region_center(region_img)
    grasp_point = tuple(grasp_center)
    grasp_angle = ang_img[grasp_point]
    g = Grasp(grasp_point, grasp_angle)
    if width_img is not None:
        g.length = width_img[grasp_point]
        g.width = g.length / 2
    return g  # 返回抓取框

def center_generate_grasps(ang_img, center, width_img=None):  # 输入中心点，返回抓取框
    grasp_point = tuple(center)
    grasp_angle = ang_img[grasp_point]
    g = Grasp(grasp_point, grasp_angle)
    if width_img is not None:
        g.length = width_img[grasp_point]
        g.width = g.length / 2
    return g  # 返回抓取框

def generate_canvas(box_region, region_img):  # 输入框四点和物体区域，绘制重叠图
    rr1, cc1 = box_region.refine_polygon_coords()#给出四点，得到组成的区域内的像素
    if np.any(rr1 > 299):
        print('rrrrrrrrrrrrrrr1>299')
        rr1 = np.where(rr1 > 299, 299, rr1)
    if np.any(cc1 > 299):
        print('ccccccccccccccc1>299')
        cc1 = np.where(cc1 > 299, 299, cc1)
    region_img1 = region_img.copy()
    canvas = region_img1  # 创建一个和region_img大小一样的画布并把region_img的值填入
    canvas[rr1, cc1] = canvas[rr1, cc1] + 1  # 重叠部分的值为2
    canvas = np.where(canvas == 1, 0, canvas)  # 将是1的部分换为0
    canvas = np.where(canvas == 2, 1, canvas)  # 将重叠部分（值为2）换为1
    return canvas  # 返回重叠图

def refine_detect_grasps(q_img, ang_img, region_img, width_img=None, no_grasps=1):
    global g, grasp_center
    grasps = []
    graspss = []
    graspsss = []
    graspssss = []
    # 得到初步抓取框
    grasp_first = detect_grasps(q_img, ang_img, width_img, no_grasps=1)  # 得到初步抓取框list

    # 如果没生成抓取框
    if len(grasp_first) == 0:
        print('first grasp box is empty')
        region_img1 = region_img.copy()
        box_region = generate_grasps(ang_img, region_img1, width_img)  # 得到以物体区域为中心的抓取框
        grasps.append(box_region)
        return grasps
    # 如果生成了初步抓取框
    else:
        print('first grasp box is not empty')
        a = grasp_first[0].as_gr  # a为初步抓取框
        region_img1 = region_img.copy()
        canvas = generate_canvas(a, region_img1)  # 得到框与物体的重叠部分
        # 无相交部分
        if np.all(canvas == 0):
            print('first grasp box is not empty but canvas is empty')
            region_img2 = region_img.copy()
            box_iou_zero = generate_grasps(ang_img, region_img2, width_img)
            graspss.append(box_iou_zero)
            return graspss
        # 有相交部分，计算重叠区域的中心点grasp_point_final
        else:
            print('first grasp box is not empty and canvas is not empty')
            cY,cX = find_region_center(canvas)
            # 有相交部分，且一开始预测的就不好
            while grasp_first[0].center != (cY,cX):
                print('first grasp box is not empty and canvas is not empty and box needs refined')
                refine_center = (cY,cX)
                grasp_first[0] = center_generate_grasps(ang_img, refine_center, width_img)  # refine过后的抓取框
                graspsss.append(grasp_first[0])  # 新的框

                new_four_point = grasp_first[0].as_gr  # 新的框的四个点
                region_img3 = region_img.copy()
                canvas = generate_canvas(new_four_point, region_img3) #新的框和物体区域计算新的交集
                cY,cX = find_region_center(canvas) #寻找新的区域的中心点
                if len(graspsss) > 10:
                    break
            # 有相交部分，且一开始预测的就很好（框的中心点在区域中心点处）
            if grasp_first[0].center == (cY,cX):
                print('successfully generate final grasp box!')
            else:
                print('there is somehow error for generating final grasp box')

            print(grasp_first[0])
            graspssss.append(grasp_first[0])
            return graspssss

###########################在这里加Refine模块#############################
# def detect_grasps(q_img, ang_img, region_img, width_img=None, no_grasps=1): #网络输出
#     """
#     Detect grasps in a network output.
#     :param q_img: Q image network output
#     :param ang_img: Angle image network output
#     :param width_img: (optional) Width image network output
#     :param no_grasps: Max number of grasps to return
#     :return: list of Grasps
#     """
#     local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
#
#     grasps = []
#     for grasp_point_array in local_max:
#         grasp_point = tuple(grasp_point_array)
#
#         grasp_angle = ang_img[grasp_point]
#
#         g = Grasp(grasp_point, grasp_angle)
#         if width_img is not None:
#             g.length = width_img[grasp_point]
#             g.width = g.length / 2
#
#         grasps.append(g)
#
#
#     return grasps