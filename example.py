from skimage.feature import peak_local_max
import cv2
import numpy as np

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1): #生成初框
    
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
    return grasp #返回列表

def find_region_center(region): #输入区域，输出区域中心点
    M = cv2.moments(region)
    if M["m00"] ==0:
        print('first grasp box is empty and there is no region')
        grasp_center=[150,150]
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        grasp_center = (cY, cX)
    return grasp_center

def generate_grasps(ang_img, region_img, width_img=None): #输入区域，返回以区域中心为中心点的抓取框
    grasp_center = find_region_center(region_img)
    grasp_point = tuple(grasp_center)
    grasp_angle = ang_img[grasp_point]
    g = Grasp(grasp_point, grasp_angle)
    if width_img is not None:
        g.length = width_img[grasp_point]
        g.width = g.length / 2
    return g  # 返回抓取框

def center_generate_grasps(ang_img, center, width_img=None): #输入中心点，返回抓取框
    grasp_point = tuple(center)
    grasp_angle = ang_img[grasp_point]
    g = Grasp(grasp_point, grasp_angle)
    if width_img is not None:
        g.length = width_img[grasp_point]
        g.width = g.length / 2
    return g  # 返回抓取框

def generate_canvas(box_region, region_img):#输入框四点和物体区域，绘制重叠图
    rr1, cc1 = box_region.refine_polygon_coords()
    if np.any(rr1 > 299):
        print('rrrrrrrrrrrrrrr1>299')
        np.where(rr1 > 299, 299, rr1)
    if np.any(cc1 > 299):
        print('ccccccccccccccc1>299')
        np.where(cc1 > 299, 299, cc1)
    region_img1 = region_img.copy()
    canvas = region_img1  # 创建一个和region_img大小一样的画布并把region_img的值填入
    canvas[rr1, cc1] += 1  # 重叠部分的值为2
    canvas = np.where(canvas == 1, 0, canvas)  # 将是1的部分换位0
    canvas = np.where(canvas == 2, 1, canvas)  # 将重叠部分（值为2）换为1
    return canvas #返回重叠图

def refine_detect_grasps(q_img, ang_img, region_img, width_img=None):
    global g, grasp_center
    grasps= []
    graspss = []
    graspsss = []
    graspssss = []
    # 得到初步抓取框
    grasp_first =detect_grasps(q_img, ang_img, width_img=None, no_grasps=1)#得到初步抓取框list

    #如果没生成抓取框
    if len(grasp_first) == 0:
        print('first grasp box is empty')
        region_img1 = region_img.copy()
        box_region = generate_grasps(ang_img, region_img1, width_img=None)#得到以物体区域为中心的抓取框
        grasps.append(box_region)
        return grasps
    #如果生成了初步抓取框
    else:
        print('first grasp box is not empty')
        a = grasp_first[0].as_gr #a为初步抓取框
        canvas = generate_canvas(a, region_img) #得到框与物体的重叠部分
        # 无相交部分
        if np.all(canvas == 0):
            print('first grasp box is not empty but canvas is empty')
            region_img2 = region_img.copy()
            box_iou_zero = generate_grasps(ang_img, region_img2, width_img=None)
            graspss.append(box_iou_zero)
            return graspss
        #有相交部分，计算重叠区域的中心点grasp_point_final
        else:
            print('first grasp box is not empty and canvas is not empty')
            grasp_center = find_region_center(canvas)
            #有相交部分，且一开始预测的就不好
            while a.center != grasp_center:
                print('first grasp box is not empty and canvas is not empty and box needs refined')
                refine_center = grasp_center
                a = center_generate_grasps(ang_img, refine_center, width_img=None)#refine过后的抓取框
                graspsss.append(a) #新的框

                a_four_point = a.as_gr #新的框的四个点
                region_img1 = region_img.copy()
                canvas = generate_canvas(a_four_point, region_img1)
                grasp_center = find_region_center(canvas)
            # 有相交部分，且一开始预测的就很好（框的中心点在区域中心点处）
            if a.center == grasp_center:
                print('generate fiinal grasp box')
            else:
                print('generate fiinal grasp box error')

            print(graspsss)

            final = graspsss[-1]
            graspsss.append(final)
            return graspsss



