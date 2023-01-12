import cppsolver as cs
import numpy as np

def predictor_analytical(mag_data, pSensor, seed_params):
    tracking_converge = True
    mag_data = mag_data.reshape(-1,3)
    mag_data[:,0] = -mag_data[:,0]
    mag_data_for_calc = (mag_data.reshape(-1))
    pSensor_for_calc = (pSensor.reshape(-1))
    new_params = cs.solve_1mag(mag_data_for_calc, pSensor_for_calc, seed_params)

    if np.abs(seed_params[4]) > 1 or seed_params[5] > 0.8 or seed_params[5] < -0.2 or seed_params[6] < -0.1 or seed_params[6] > 0.7:
        # print("OUT OF RANGE")
        tracking_converge = False
    elif np.linalg.norm(seed_params[4:7]-new_params[4:7]) < 0.1:
        tracking_converge = True
    else:
        # print("POSITION LOSS")
        tracking_converge = False

    location_data = [new_params[4] * 1e2, new_params[5] * 1e2, new_params[6] * 1e2]

    return tracking_converge, location_data, new_params