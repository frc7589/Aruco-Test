import numpy as np
from networktables import NetworkTables
import cv2
import cv2.aruco as aruco
import time
import math
from numpy.linalg import inv

if __name__ == '__main__': 

    # roborio_ip = 'roborio-7589-frc.local'
    # NetworkTables.initialize(server=roborio_ip)
    # sd = NetworkTables.getTable('SmartDashboard')
    one = 0
    two = 0.1
    three = 0.1
    c_in_w = np.array([],dtype="double")
    carmera_original_point = np.array([[0],
                                    [0],
                                    [0]],dtype="double")
    carmera_forward_point = np.array([[0.01],
                                    [0],
                                    [0]],dtype="double")
    image_points = np.array([], dtype="double")
    debug = True
    detect = []
    position = []
    average_x = []
    average_y = []
    average_z = []
    a = 0
    aver_x = 0
    aver_y = 0
    aver_z = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    mistake = [42,62,111]
    rotation_vector = np.zeros((3,1))
    translation_vector = np.zeros((3,1)) 
    objPoints = np.array([[0,0,0],
                        [0.08,0, 0],
                        [0.08,0.1, 0],
                        [0,0.08, 0],#0
                        [0.1,0,0],
                        [0.18,0, 0],
                        [0.18,0.08, 0],
                        [0.1,0.08, 0],#1
                        [0,0.1,0],
                        [0.08,0.1, 0],
                        [0.08,0.18, 0],
                        [0,0.18, 0],#2
                        [0.1,0.1,0],
                        [0.18,0.1, 0],
                        [0.18,0.18, 0],
                        [0.1,0.18, 0]#3
                        ], dtype=np.float64)    
    camera_matrix = np.array(
                            [[829.5010500,0,312.263367],
                            [0,828.729047,207.158177],
                            [0, 0, 1]], dtype = np.float64)
    dist_coeffs = np.zeros((5,1)) 
    cap = cv2.VideoCapture(1)

    while(True):
        ret, frame = cap.read()
        debug = True
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        arucoParameters = aruco.DetectorParameters_create()

        corners0, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=arucoParameters)

        frame = aruco.drawDetectedMarkers(frame, corners0,ids)
        
        #cv2.imshow('Display',frame)
        
        if np.all(ids != None):
            # for i in range(ids.size):
            #     if ids[i][0]>3:
            #         debug = False
            corners_new = cv2.cornerSubPix(frame,1,(5,5),(-1,-1),criteria)
            # if not debug:
            #     continue

            # for i in range(ids.size):
            #     detect.append(ids[i][0])

            # for i in range(ids.size):
            #     if a == 0:
            #         x1_left_down = [corners0[i][0][0][0], corners0[i][0][0][1]]
            #         x2_right_down = [corners0[i][0][1][0], corners0[i][0][1][1]]
            #         x3_left_up = [corners0[i][0][2][0], corners0[i][0][2][1]]
            #         x4_right_up = [corners0[i][0][3][0], corners0[i][0][3][1]]
            #         pts_dst = np.array([x1_left_down,x2_right_down, x3_left_up,x4_right_up])
            #         a+=1

            #     else:
            #         for j in range(4):
            #             x1_left_down = [corners0[i][0][j][0], corners0[i][0][j][1]]
            #             x2_right_down = [x1_left_down]
            #             pts_dst = np.append(pts_dst,x2_right_down,axis=0)
            
            # objPoints_in = np.array(objPoints[detect[0]*4:detect[0]*4+4])

            # for i in range(1,ids.size):
            #     objPoints_in = np.append(objPoints_in,objPoints[detect[i]*4:detect[i]*4+4],axis=0)
           
            # (success, rotation_vector, translation_vector) = cv2.solvePnP(objPoints_in,pts_dst, camera_matrix, None)

            # frame = aruco.drawAxis(frame,camera_matrix,dist_coeffs,rotation_vector,translation_vector,0.1)

            # R_back, _ = cv2.Rodrigues(rotation_vector)
            
            # invert_R = inv(R_back)

            
            # R = carmera_original_point - translation_vector

            # R_F = carmera_forward_point - translation_vector

            # c_in_w = np.dot(invert_R,R)

            # print(c_in_w)
            # average_x.append(c_in_w[0][0])
            # average_y.append(c_in_w[1][0])
            # average_z.append(c_in_w[2][0])
            # if len(average_x) == 5:
            #     for i in range(5):
            #         aver_x  += average_x[i]
            #         aver_y += average_y[i]
            #         aver_z += average_z[i]
            #     aver_x /= 5
            #     aver_y /= 5
            #     aver_z /= 5
            # # for i in range(3):
            # #     c_in_w[i] = c_in_w[i]*100
            # print(c_in_w)
            # for i in range(3):
            #     pos[i][a] = c_in_w[i]
            # x = c_in_F[0] - c_in_w[0]
            # y = c_in_F[1] - c_in_w[1]
            # sita = math.degrees(math.atan(y/x))
            # a+=1
            # if a==5:
            #     for i in range(3):
            #         new = ((pos[i][0]+pos[i][1]+pos[i][2]+pos[i][3]+pos[i][4])/5*100)-mistake[i]
            #         print("{:.1f}".format(new),end=" ")
            #         if i==2:
            #             print("")
            #     a = 0
            # print(sita)
            # "{:.1f}".format(new),end=""
          
            
            # sd.putNumber('cord_x', float(c_in_w[0]))
            # sd.putNumber('cord_y', float(c_in_w[1]))
            # sd.putNumber('cord_z', float(c_in_w[2]))
            # sd.putNumber('cord_sita', float(sita))
            a = 0
            detect.clear()
        cv2.imshow('Display',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    