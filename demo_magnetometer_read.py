import serial
import serial.tools.list_ports
import binascii
import time
import numpy as np
import csv

class mangnetometerArray(object):

    def __init__(self, arrayPort=None, arrayBaudrate=None):
        self.mag_data = np.array((7,16,3))
        self.mag_line_data_read = np.array(7) 
        self.ser = serial.Serial(port=arrayPort, baudrate=arrayBaudrate)
        self.startTime = time.time_ns()
        self.lastTime = time.time_ns()
        time.sleep(5)
        self.ser.flushInput()
        self.ser.flush()
        self.ser.write(binascii.unhexlify('FFFFFFFFFFFFFFFF'))
        self.ser.read_until(binascii.unhexlify('525D'))
        self.ser.write(binascii.unhexlify('FFFFFFFFFFFFFFFF'))
        print(f'COM Port: {arrayPort} initialized')
        while(self.ser.inWaiting()):
            garbage_data = self.ser.read(self.ser.inWaiting())

    def readWriteMeasurement(self, magnetometer_data_dict):
        num = 16
        # global magnetometer_data_dict
        # global all_lines_data_read
        sensors = np.zeros((num, 3))

        while(self.ser.inWaiting() < 136):
            time.sleep(0.001)
        self.ser.read_until(binascii.unhexlify('424D'))
        dataframe = self.ser.read(134)
        serData = dataframe[0:1]
        self.arrayID = int.from_bytes(serData, byteorder='big')
        line_index = 6 - self.arrayID
        serData = dataframe[1:2]
        self.num_sensors = int.from_bytes(serData, byteorder='big')
        serData = dataframe[2:6]
        self.sampleTimeOffsetUS = int.from_bytes(serData, byteorder='big')
        for sensorNumber in range(self.num_sensors):
            local_checksum = 0
            sensorXAxisData = int.from_bytes(
                dataframe[6+(sensorNumber*8): 8+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorXAxisData)
            sensorYAxisData = int.from_bytes(
                dataframe[8+(sensorNumber*8): 10+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorYAxisData)
            sensorZAxisData = int.from_bytes(
                dataframe[10+(sensorNumber*8): 12+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorZAxisData)
            checksum = int.from_bytes(
                dataframe[12+(sensorNumber*8): 14+(sensorNumber*8)], byteorder='big', signed=True)
            if(checksum == (local_checksum & 0xFFFF)):
                sensors[sensorNumber, 0] = sensorXAxisData
                sensors[sensorNumber, 1] = sensorYAxisData
                sensors[sensorNumber, 2] = sensorZAxisData

        sensors = sensors.reshape(-1)

        if line_index == 0:
            magnetometer_data_dict['line 1'] = sensors
            magnetometer_data_dict['line1_data_read'] = True
        elif line_index == 1:
            magnetometer_data_dict['line 2'] = sensors
            magnetometer_data_dict['line2_data_read'] = True
        elif line_index == 2:
            magnetometer_data_dict['line 3'] = sensors
            magnetometer_data_dict['line3_data_read'] = True
        elif line_index == 3:
            magnetometer_data_dict['line 4'] = sensors
            magnetometer_data_dict['line4_data_read'] = True
        elif line_index == 4:
            magnetometer_data_dict['line 5'] = sensors
            magnetometer_data_dict['line5_data_read'] = True
        elif line_index == 5:
            magnetometer_data_dict['line 6'] = sensors
            magnetometer_data_dict['line6_data_read'] = True
        elif line_index == 6:
            magnetometer_data_dict['line 7'] = sensors
            magnetometer_data_dict['line7_data_read'] = True
        else:
            print(str(line_index)+" Warning: Data Read ERROR!")

def collect_magnetometer_background_data(magnetometer_data_dict, num_datapoints, csv_file_writer):

    while len(magnetometer_data_dict) < 2 * 7:
        time.sleep(0.001)

    print("Please remove all magnetic elements from vicinty of MagDesk")
    background_noise = np.zeros((num_datapoints, (7*16*3)))
    time.sleep(10)
    for background_noise_writer_iter in range(num_datapoints):
        while(not(magnetometer_data_dict['line1_data_read'] and magnetometer_data_dict['line2_data_read'] and magnetometer_data_dict['line3_data_read']
         and magnetometer_data_dict['line4_data_read'] and magnetometer_data_dict['line5_data_read']  and magnetometer_data_dict['line6_data_read']
          and magnetometer_data_dict['line7_data_read'])):
            time.sleep(0.001)
        
        magdesk_line_1 = magnetometer_data_dict['line 1']
        magdesk_line_2 = magnetometer_data_dict['line 2']
        magdesk_line_3 = magnetometer_data_dict['line 3']
        magdesk_line_4 = magnetometer_data_dict['line 4']
        magdesk_line_5 = magnetometer_data_dict['line 5']
        magdesk_line_6 = magnetometer_data_dict['line 6']
        magdesk_line_7 = magnetometer_data_dict['line 7']
        magdesk_data = np.concatenate(
            (magdesk_line_1, magdesk_line_2, magdesk_line_3, magdesk_line_4, magdesk_line_5, magdesk_line_6, magdesk_line_7), axis=0)
        magdesk_data = magdesk_data.reshape(-1) 
        # magdesk_data_list = np.ndarray.tolist(magdesk_data)
        
        background_noise[background_noise_writer_iter] = magdesk_data
        print(f"Number of background datapoints collected: {background_noise_writer_iter} / {num_datapoints}", end = "\r")

        # csv_row = [time.time_ns()] + magdesk_data_list
        # csv_file_writer.writerow(csv_row)
        
        magnetometer_data_dict['line1_data_read'] = False
        magnetometer_data_dict['line2_data_read'] = False
        magnetometer_data_dict['line3_data_read'] = False
        magnetometer_data_dict['line4_data_read'] = False
        magnetometer_data_dict['line5_data_read'] = False
        magnetometer_data_dict['line6_data_read'] = False
        magnetometer_data_dict['line7_data_read'] = False
    
    return 1, np.mean(background_noise, axis=0), np.std(background_noise, axis=0)

def read_all_rows(magnetometer_data_dict):
    while(not(magnetometer_data_dict['line1_data_read'] and magnetometer_data_dict['line2_data_read'] and magnetometer_data_dict['line3_data_read']
         and magnetometer_data_dict['line4_data_read'] and magnetometer_data_dict['line5_data_read']  and magnetometer_data_dict['line6_data_read']
          and magnetometer_data_dict['line7_data_read'])):
            time.sleep(0.001)
    
    magdesk_line_1 = magnetometer_data_dict['line 1']
    magdesk_line_2 = magnetometer_data_dict['line 2']
    magdesk_line_3 = magnetometer_data_dict['line 3']
    magdesk_line_4 = magnetometer_data_dict['line 4']
    magdesk_line_5 = magnetometer_data_dict['line 5']
    magdesk_line_6 = magnetometer_data_dict['line 6']
    magdesk_line_7 = magnetometer_data_dict['line 7']
    magdesk_data = np.concatenate(
        (magdesk_line_1, magdesk_line_2, magdesk_line_3, magdesk_line_4, magdesk_line_5, magdesk_line_6, magdesk_line_7), axis=0)
    magdesk_data = magdesk_data.reshape(-1) 

    magnetometer_data_dict['line1_data_read'] = False
    magnetometer_data_dict['line2_data_read'] = False
    magnetometer_data_dict['line3_data_read'] = False
    magnetometer_data_dict['line4_data_read'] = False
    magnetometer_data_dict['line5_data_read'] = False
    magnetometer_data_dict['line6_data_read'] = False
    magnetometer_data_dict['line7_data_read'] = False

    return magdesk_data