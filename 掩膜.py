# Forest
import cv2
import numpy as np
import camera_configs_YM  # 摄像头的标定数据

cap=cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)#camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)#camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

def depth(val = 0):
    window_size = 3
    min_disp = 0
    num_disp = 128 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 50
    disp12MaxDiff = 12
    P1 = 72
    P2 = 288
    cv2.namedWindow('disparity')
    stereo = cv2.StereoSGBM_create(
        minDisparity= min_disp,
        numDisparities= num_disp,
        blockSize= window_size,
        uniquenessRatio= uniquenessRatio,
        speckleRange= speckleRange,
        speckleWindowSize= speckleWindowSize,
        disp12MaxDiff= disp12MaxDiff,
        P1= P1,
        P2= P2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = stereo.compute(img1_rectified,img2_rectified).astype(np.float32)/16.0
    cv2.imshow('left',img1_rectified)
    cv2.imshow('disparity',(disp-min_disp)/num_disp)
    # print(camera_configs_YM.Q)
    points_3d = cv2.reprojectImageTo3D(disp, camera_configs_YM.Q)
    print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x_3, y_3, points_3d[y_3, x_3, 0], points_3d[y_3, x_3, 1], points_3d[y_3, x_3, 2]))

while(1):
    # 获取每一帧
    ret,frame=cap.read()

    frame1 = frame[0:480, 0:640]#frame2 = frame[0:480, 0:640]#此时1对应C1,L,相机坐标系起点
    frame2 = frame[0:480, 640:1280]#frame1 = frame[0:480, 640:1280]

    # 根据标定数据对图片进行重构消除图片的畸变
    img1_rectified = cv2.remap(frame1, camera_configs_YM.left_map1, camera_configs_YM.left_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(frame2, camera_configs_YM.right_map1, camera_configs_YM.right_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)

    # remap()的图是反的
    # img1_rectified = cv2.flip(img1_rectified, -1)
    # img2_rectified = cv2.flip(img2_rectified, -1)

    #转换到HSV
    hsv1=cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
    hsv2=cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2HSV)
    #设定绿色阈值
    # lower_green1 = np.array([45, 140, 75])
    # upper_green1 = np.array([70, 230, 125])
    #
    # lower_green2 = np.array([55, 170, 80])
    # upper_green2 = np.array([70, 240, 130])
    lower_green1 = np.array([45, 100, 70])
    upper_green1 = np.array([70, 230, 150])

    lower_green2 = np.array([45, 100, 70])
    upper_green2 = np.array([70, 240, 150])

    #根据阈值构建掩膜
    mask1=cv2.inRange(hsv1,lower_green1,upper_green1)
    mask2=cv2.inRange(hsv2,lower_green2,upper_green2)

    # 均值滤波
#    img_blur1= cv2.blur(mask1, (5, 5))
#    img_blur2= cv2.blur(mask2, (5, 5))
    #高斯滤波
#    img_blur1 = cv2.GaussianBlur(mask1, (3, 3), 9)
#    img_blur2 = cv2.GaussianBlur(mask2, (3, 3), 9)
    #先腐蚀再膨胀
    kernel = np.ones((3, 3), np.uint8)
    erosion1 = cv2.erode(mask1, kernel, iterations=1)
    erosion2 = cv2.erode(mask2, kernel, iterations=1)
    kernel2 = np.ones((3, 3), np.uint8)
    dilation1 = cv2.dilate(erosion1, kernel, iterations=10)
    dilation2 = cv2.dilate(erosion2, kernel, iterations=10)

    #对原图像和掩膜进行位运算
#    res1=cv2.bitwise_and(frame1,frame1,mask=dilation1)
#    res2=cv2.bitwise_and(frame2,frame2,mask=dilation2)



    contours1,hierarchy1 = cv2.findContours(dilation1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2,hierarchy2 = cv2.findContours(dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_3, y_3 = 0, 0

    for c1 in contours1:
        # find bounding box coordinates
        x1, y1, w, h = cv2.boundingRect(c1)
        cv2.rectangle(img1_rectified, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        x_3,y_3=int(x1+0.5*w),int(-(y1+0.5*h))


    for c2 in contours2:
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c2)
        cv2.rectangle(img2_rectified, (x, y), (x + w, y + h), (0, 255, 0), 2)

#    contours, hierarchy = cv2.findContours(dilation1, 1, 2)
#    cnt = contours[0]
#    (x, y, w, h) = cv2.boundingRect(cnt)
#    img1=cv2.rectangle(dilation1, (x, y), (x + w, y + h), (255, 0, 0), 3)

    #显示
    cv2.imshow('frame1',img1_rectified)
#    cv2.imshow('mask1',dilation1)
#    cv2.imshow('res1',res1)
    cv2.imshow('frame2',img2_rectified)
#    cv2.imshow('mask2',dilation2)
#    cv2.imshow('res2',res2)

    depth()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 结束当前程序
        break
    #关闭窗口
cap.release()
cv2.destroyAllWindows()





