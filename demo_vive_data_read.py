import triad_openvr
import numpy as np
import math
import time
from os import path
from scipy.spatial.transform import Rotation as R

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

class vive_tracker(object):
    def __init__(self, recalib=None, calib_file=None, tracking_distance=None, controller_id=None):
        #openVR tracking object
        self.vive = triad_openvr.triad_openvr()
    
        #Flag for whether calibration variables hold information
        self.calib_done = False
        #Set to true if you want to recalibrate on every run
        self.recalib = recalib
        #Filename of vive calibration information
        self.vive_calib_file_name = calib_file

        #Variables for holding Vive calibration information
        self.calib_rotation_matrix = np.zeros((3,3))
        self.calib_translation_matrix = np.zeros(3)

        if (path.exists(self.vive_calib_file_name) and (not(self.recalib))):
            self.calib_rotation_matrix = np.load(self.vive_calib_file_name)['rotation']
            self.calib_translation_matrix = np.load(self.vive_calib_file_name)['translation']
            self.calib_done = True

        #Distance in between Vive tracking point and mounted magnet
        self.vive_tracking_distance = tracking_distance

        #Controller ID e.g. "controller_1" / "controller_2"
        self.controller_id = controller_id

        #controller location data
        self.location_x = 0
        self.location_y = 0
        self.location_z = 0

        #controller quarternions data
        self.r_w = 0
        self.r_x = 0
        self.r_y = 0
        self.r_z = 0

        #controller euler angle data
        self.pitch = 0
        self.yaw = 0
        self.roll = 0

        #controller data ready to read
        self.ready_to_read = False

    def update_pose_data(self):
        #Use open_vr to get the position+rotation matrix
        position_rotation_matrix = self.vive.devices[self.controller_id].get_pose_matrix()

        #convert list based data structure to numpy array
        position_rotation_matrix = np.asmatrix(list(position_rotation_matrix))

        #get location data
        x = position_rotation_matrix[0,3]
        y = position_rotation_matrix[1,3]
        z = position_rotation_matrix[2,3]

        location_data = np.array([[float(x)], [float(y)], [float(z)]])

        #Get the rotation matrix    
        rotation_matrix = position_rotation_matrix[:, 0:3]

        #Apply calibration only if variables hold valid info
        if(self.calib_done):
            #apply calibration to rotation matrix
            rotation_matrix = self.calib_rotation_matrix @ rotation_matrix
        
        #calculate quarternions
        r_w = math.sqrt(abs(1+rotation_matrix[0,0]+rotation_matrix[1,1]+rotation_matrix[2,2]))/2
        r_x = (rotation_matrix[2,1]-rotation_matrix[1,2])/(4*r_w)
        r_y = (rotation_matrix[0,2]-rotation_matrix[2,0])/(4*r_w)
        r_z = (rotation_matrix[1,0]-rotation_matrix[0,1])/(4*r_w)

        quarternions_data = np.array([[float(r_w)], [float(r_x)], [float(r_y)], [float(r_z)]])

        #calculate euler angles
        yaw = 180 / math.pi * math.atan2(rotation_matrix[1,0], rotation_matrix[0, 0])
        roll = 180 / math.pi * math.atan2(rotation_matrix[2,0], rotation_matrix[0, 0])
        pitch = 180 / math.pi * math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        euler_angle_data = np.array([[float(pitch)], [float(yaw)], [float(roll)]])

        #Apply calibration only if variables hold valid info
        if(self.calib_done):
            location_data = (self.calib_rotation_matrix@location_data) + self.calib_translation_matrix

            if(np.abs(roll) > 65 and np.abs(roll) < 115):
                self.ready_to_read = False
                return

            pitch_radians = math.radians(pitch -  90) 
            yaw_radians = math.radians(yaw) 

            location_data[0] = float(location_data[0] + (self.vive_tracking_distance * math.cos(pitch_radians) * math.sin(yaw_radians)))
            location_data[1] = float(location_data[1] - (self.vive_tracking_distance * math.cos(pitch_radians) * math.cos(yaw_radians)))
            location_data[2] = float(location_data[2] - (self.vive_tracking_distance * math.sin(pitch_radians)))

        #Set ready to read False
        self.ready_to_read = False

        #controller location data
        self.location_x = location_data[0,0]
        self.location_y = location_data[1,0]
        self.location_z = location_data[2,0]

        #controller quarternions data
        self.r_w = quarternions_data[0,0]
        self.r_x = quarternions_data[1,0]
        self.r_y = quarternions_data[2,0]
        self.r_z = quarternions_data[3,0]

        #controller euler angle data
        self.pitch = euler_angle_data[0,0]
        self.yaw = euler_angle_data[1,0]
        self.roll = euler_angle_data[2,0]

        #Set ready to read False
        self.ready_to_read = True

    def calibrate(self, gt_calib_location_data, gt_test_location_data):
        #Do not calibrate if calibration already done!
        if self.calib_done == True:
            return
        
        vive_calib_data = np.zeros([3, np.shape(gt_calib_location_data)[1]])
        vive_calib_data_test = np.empty([3,1])

        #Collect data from calibration points
        for iter in range(np.shape(gt_calib_location_data)[1]):
            print(f"Press Vive Controller System button when you have kept the controller at static location {iter + 1}")
            # state_data_raw = self.vive.devices[self.controller_id].get_controller_inputs()
            state_data = self.vive.devices[self.controller_id].get_controller_inputs()
            while state_data['trackpad_pressed'] == False:
                state_data = self.vive.devices[self.controller_id].get_controller_inputs()
            self.update_pose_data()
            vive_calib_data[0,iter] = self.location_x
            vive_calib_data[1,iter] = self.location_y
            vive_calib_data[2,iter] = self.location_z
            self.vive.devices[self.controller_id].trigger_haptic_pulse(100000)
            time.sleep(2)

        #Compute rotation and translation matrix
        print("Calibrating...", end = '')
        self.calib_rotation_matrix, self.calib_translation_matrix = rigid_transform_3D(vive_calib_data, gt_calib_location_data)
        self.calib_done = True
        print(" Done!")

        #Testing Calibration
        print("Testing Calibration...")
        print(f"Press Vive Controller System button when you have kept the controller at the test location")
        state_data = self.vive.devices[self.controller_id].get_controller_inputs()
        while state_data['trackpad_pressed'] == False:
            state_data = self.vive.devices[self.controller_id].get_controller_inputs()
        self.update_pose_data()
        vive_calib_data_test[0,0] = self.location_x
        vive_calib_data_test[1,0] = self.location_y
        vive_calib_data_test[2,0] = self.location_z
        print(vive_calib_data_test)
        self.vive.devices[self.controller_id].trigger_haptic_pulse()

        # RMSE calculation
        err = (vive_calib_data_test - gt_test_location_data) * 100
        err = err * err
        err = np.sum(err)
        rmse_test_data = np.sqrt(err)
        print(f'RMSE in calibration @ Test datapoint: {rmse_test_data}')
        
        #Save a copy of calibration parameters for next run
        np.savez('vive_calib', rotation=self.calib_rotation_matrix, translation=self.calib_translation_matrix)