#Import Stuffs
from re import sub
from codetiming import Timer
import multiprocessing
import time
import numpy as np
import serial
import serial.tools.list_ports
import csv
from datetime import datetime
from filterpy.kalman import FixedLagSmoother
from scipy import ndimage
from config import pSensor_7_line_elevated_z_1cm_below_table
import demo_vive_data_read as gt_tracker
import demo_plotting as plot_tracking
from demo_magnetometer_read import mangnetometerArray, collect_magnetometer_background_data, read_all_rows
from demo_predictor_analytical import predictor_analytical

#Sensor Arrangement
pSensor = pSensor_7_line_elevated_z_1cm_below_table

#Feather Baud Rate
BAUDRATE = 921600

#MagDesk Config
ROW_MAGNETOMETERS = 7
COLUMN_MAGNETOMETERS = 16
NUM_AXES_MAGNETOMETER = 3

#Featurization Settings
ML_MAGNETOMETER_ARRAY_SQUARE_SIZE = 3
ML_MAGNETOMETER_ARRAY_XY_FEATURE_LEN = 2

# Prediction Modes:
# 0 : Analaytical Mode
# 1 : ML Mode
# 2 : Hybrid Mode
prediction_mode = multiprocessing.Value("i", 0)

analytical_params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(2.2), 1e-2 * -75, 1e-2 * (17.5), 1e-2 * (7), np.pi, 0])

#Background Noise number of data points
NUM_BACKGROUND_DATAPOINTS = 1000

filename_timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M_%p")
magnet_name = 'SA'
background_noise_dataset_filename = 'data/ml_dataset/magdesk_dataset_background_noise_' + magnet_name + "_" + filename_timestamp + '.csv'
background_noise_dataset_file = open(background_noise_dataset_filename, mode='w')
background_noise_dataset_writer = csv.writer(background_noise_dataset_file, delimiter=',')

#Number of data points to plot 
NUM_DATA_POINTS_TO_SHOW_PATH = 300

#Random Forest Model Filename
rf_model_filename = 'voxel_1cm_9_mag_03_05_new_table_sec1_4.joblib'

#ML Model to use
model_filename = rf_model_filename

#Variables which would store plotting data
magnetometer_data_dict = multiprocessing.Manager().dict()
magdesk_data_dict = multiprocessing.Manager().dict()
vive_data_dict = multiprocessing.Manager().dict()
predicted_path_data = multiprocessing.Manager().list()
gt_path_data = multiprocessing.Manager().list()
background_data_collection_done = multiprocessing.Value("i", 0)
background_data_mean = multiprocessing.Manager().dict()
background_data_std = multiprocessing.Manager().dict()
tracker_calibration_done = multiprocessing.Value("i", 0)
num_datapoints_to_show = multiprocessing.Value("i", NUM_DATA_POINTS_TO_SHOW_PATH)
kalman_filter_use = multiprocessing.Value("i", 1)

#ground truth tracker object
gt_data_reader = None
#Set to true if you want to recalibrate on every run
VIVE_RECALIB = True
#Filename of vive calibration information
vive_calib_file_name = 'vive_calib.npz'
#Distance in between Vive tracking point and mounted magnet
# vive_magdesk_tracking_point_differences = 18 * 1e-2
vive_magdesk_tracking_point_differences = 19.5 * 1e-2
#Vive controller ID
vive_controller_id = "controller_1"

def read_ground_truth_data():
    global gt_data_reader
    global vive_data_dict
    global tracker_calibration_done

    vive_ground_truth_data = np.array([[-86, 86, 86],[19, 19, 79.2],[6.0, 6.0, 6.0]])
    vive_ground_truth_data = vive_ground_truth_data * 1e-2
    # vive_ground_truth_test = np.array([[-69], [92.5], [5]])
    vive_ground_truth_test = np.array([[-67.5], [92.5], [4.5]])
    vive_ground_truth_test = vive_ground_truth_test * 1e-2


    gt_data_reader = gt_tracker.vive_tracker(VIVE_RECALIB, vive_calib_file_name, 
        vive_magdesk_tracking_point_differences, vive_controller_id)
    gt_data_reader.calibrate(vive_ground_truth_data, vive_ground_truth_test)
    tracker_calibration_done.value = 1
    while True:
        gt_data_reader.update_pose_data()
        if gt_data_reader.ready_to_read == True:
            vive_data_dict['x'] = gt_data_reader.location_x
            vive_data_dict['y'] = gt_data_reader.location_y
            vive_data_dict['z'] = gt_data_reader.location_z

def readArray(arrayPort=None):
    global magnetometer_data_dict
    print(arrayPort)
    magnetometerArrayVar = mangnetometerArray(
        arrayPort=arrayPort, arrayBaudrate=BAUDRATE)
    while True:
        magnetometerArrayVar.readWriteMeasurement(magnetometer_data_dict)

def read_magdesk():
    global magdesk_data_dict
    global magnetometer_data_dict
    global background_data_collection_done
    global background_data_mean

    while len(magnetometer_data_dict) < 14 or background_data_collection_done.value == 0:
        time.sleep(0.001)
    
    while True:
        magdesk_data_dict["data"] = read_all_rows(magnetometer_data_dict) - background_data_mean["data"]

def collect_background_data():
    global magnetometer_data_dict
    global background_data_collection_done
    global background_data_mean
    global background_data_std
    global tracker_calibration_done

    while len(magnetometer_data_dict) < 14 or tracker_calibration_done.value == 0:
        time.sleep(0.001)
    background_data_collection_done.value, background_data_mean["data"], background_data_std["data"] = collect_magnetometer_background_data(magnetometer_data_dict, NUM_BACKGROUND_DATAPOINTS, background_noise_dataset_writer)

def pose_predictor(magdesk_data):
    global prediction_mode
    global analytical_params
    global background_data_std
    location_data = [0, 0, 0]
    if prediction_mode.value == 0:
        tracking_converge, location_data, analytical_params_new = predictor_analytical(magdesk_data, pSensor, analytical_params)
    return location_data, tracking_converge

def collect_update_datapoints():
    # global magnetometer_data_dict
    global magdesk_data_dict
    global vive_data_dict
    global predicted_path_data
    global gt_path_data
    global background_data_collection_done
    global analytical_params

    while len(vive_data_dict) == 0 or background_data_collection_done.value == 0 or len(magdesk_data_dict) == 0:
        time.sleep(0.001)
    
    while True:  
        magdesk_data = magdesk_data_dict["data"]
        gt_path_data.append(np.reshape(np.array([vive_data_dict['x'], vive_data_dict['y'], vive_data_dict['z']]),-1))
        analytical_params[4] = vive_data_dict['x']
        analytical_params[5] = vive_data_dict['y'] 
        analytical_params[6] = vive_data_dict['z']
        # tracking_working = True
        predicted_pose, tracking_working = pose_predictor(magdesk_data)
        if(tracking_working):
            predicted_path_data.append(np.reshape(np.array([predicted_pose[0], predicted_pose[1], predicted_pose[2]]),-1))
            # predicted_path_data.append(np.reshape(np.array([vive_data_dict['x'], vive_data_dict['y'], vive_data_dict['z']]),-1))
            # gt_path_data.append(np.reshape(np.array([vive_data_dict['x'], vive_data_dict['y'], vive_data_dict['z']]),-1))
        # gt_path_data.append(np.reshape(np.array([vive_data_dict['x'], vive_data_dict['y'], vive_data_dict['z']]),-1))
        # print(f"{gt_path_data[-1]}", end="\r")
        # time.sleep(0.01)

def plotter():
    global predicted_path_data
    global gt_path_data
    global num_datapoints_to_show
    global kalman_filter_use
    while len(gt_path_data) < 10 or len(predicted_path_data) < 10:
        time.sleep(0.001)
    plot_object = plot_tracking.plotter(pSensor)
    while True:
        kalman_filter_use.value = plot_object.kalman_on
        num_datapoints_to_show.value = plot_object.num_datapoints_plot
        if len(gt_path_data) > num_datapoints_to_show.value:
            gt_path_data_to_pass = np.array(gt_path_data[-num_datapoints_to_show.value:])
        else:
            gt_path_data_to_pass = np.array(gt_path_data)

        if len(predicted_path_data) > num_datapoints_to_show.value:
            predicted_path_data_to_pass = np.array(predicted_path_data[-num_datapoints_to_show.value:])
        else:
            predicted_path_data_to_pass = np.array(predicted_path_data)
        
        if kalman_filter_use.value == 1:
        
            fls_x = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

            fls_x.x = np.array([0., .5])
            fls_x.F = np.array([[1.,1.],
                            [0.,1.]])

            fls_x.H = np.array([[1.,0.]])
            fls_x.P *= 200
            fls_x.R *= 5.
            fls_x.Q *= 0.001

            for x in predicted_path_data_to_pass[:,0]:
                fls_x.smooth(x)
            
            x_smooth = np.array(fls_x.xSmooth)[:, 0]

            fls_y = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

            fls_y.x = np.array([0., .5])
            fls_y.F = np.array([[1.,1.],
                            [0.,1.]])

            fls_y.H = np.array([[1.,0.]])
            fls_y.P *= 200
            fls_y.R *= 5.
            fls_y.Q *= 0.001

            for y_iter in predicted_path_data_to_pass[:,1]:
                fls_y.smooth(y_iter)
            y_smooth = np.array(fls_y.xSmooth)[:, 0]

            fls_z = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

            fls_z.x = np.array([0., .5])
            fls_z.F = np.array([[1.,1.],
                            [0.,1.]])

            fls_z.H = np.array([[1.,0.]])
            fls_z.P *= 200
            fls_z.R *= 5.
            fls_z.Q *= 0.001

            for z in predicted_path_data_to_pass[:,2]:
                fls_z.smooth(z)
            z_smooth = np.array(fls_z.xSmooth)[:, 0]

            # predicted_path_data_to_pass[-30:,0] = x_smooth
            # predicted_path_data_to_pass[-30:,1] = y_smooth
            # predicted_path_data_to_pass[-30:,2] = z_smooth

            gt_path_data_to_pass = gt_path_data_to_pass[:-8]
            
            plot_object.plot(gt_path_data_to_pass, np.column_stack((x_smooth, y_smooth, z_smooth)))
        else:
            plot_object.plot(gt_path_data_to_pass, predicted_path_data_to_pass)
        # print(f"{predicted_path_data_to_pass[-1]}", end="\r")

def main():
    input("Press Enter to start the demo!")

    ports = serial.tools.list_ports.grep('ttyUSB')

    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        multiprocessing.Process(
            target=read_ground_truth_data, args=()).start()
        processes = list()
        for count, value in enumerate(ports):
            processes.append(multiprocessing.Process(
                target=readArray, args=(value.device,)))

        processes.append(multiprocessing.Process(target=collect_background_data))
        processes.append(multiprocessing.Process(target=read_magdesk))
        processes.append(multiprocessing.Process(target=collect_update_datapoints))
        processes.append(multiprocessing.Process(target=plotter))

        for iter in range(len(processes)):
            processes[iter].start()

        for iter in range(len(processes)):
            processes[iter].join()

if __name__ == '__main__':
    main()