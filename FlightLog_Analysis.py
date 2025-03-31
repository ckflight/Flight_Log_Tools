import binascii
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import scipy.signal as signal

import peakutils
from plotWindow import plotWindow

from math import factorial

# This code uses python 3.11.0 - 11

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2
    # precompute coefficients

    b = np.asmatrix([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])

    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def FindPeak_Frequency(peaks):
    # This method finds the frequency of the peaks.
    # Peak returns the array location where peaks occur.
    # To find the peak_freq, these locations in the timeArray is needed.
    # Difference between two timeArray result will show the period of the peak.

    freq_array = []
    freqArray_index = 0
    size = len(peaks)

    for i in range(size - 1):

        if (i < size - 2):
            peak1 = timeArray[peaks[i]]
            peak2 = timeArray[peaks[i + 1]]

            period = peak2 - peak1
            freq = 1 / period
            freq_array.append(freq)
            freqArray_index += 1

    return freq_array, len(freq_array)


def FFT_Calculate(array, SAMPLE_PERIOD):
    # Calculate fft of array
    rawFFT = fft(array)

    # Number of samples
    N = len(array)

    # Freq values array from 0 to 4KHz
    # Python 3 does not convert float to int auto so i use // for integer result
    freq = np.linspace(0.0, (1.0 // (2.0 * SAMPLE_PERIOD)), N // 2)

    finalFFT = (2.0 / N) * np.abs(rawFFT[0:int(N / 2)])

    peakIndexes = peakutils.indexes(finalFFT, 0.025, min_dist=50, thres_abs=True)
    if (len(peakIndexes) == 0):
        peakIndexes = np.zeros(1)

    return freq, finalFFT, peakIndexes


def StepResponse2(SP, GY, lograte=1, Ycorrection=True):
    """
    Computes the step response from quadcopter gyro data.

    Parameters:
    - SP: array-like, Setpoint signal
    - GY: array-like, Gyro response signal
    - lograte: float, Log rate in Hz (samples per second)
    - Ycorrection: bool, Whether to correct steady-state response

    Returns:
    - step_response: Averaged step response over detected step events
    - t: Time vector in milliseconds
    """
    min_input = 20  # Minimum step size to consider
    segment_length = int(lograte * 3)  # 3-second segments
    step_resp_duration_ms = 500  # Max step response duration (ms)
    t = np.arange(0, step_resp_duration_ms, 1000 / lograte)  # Time vector in ms

    # Identify step changes
    step_indices = np.where(np.abs(np.diff(SP)) >= min_input)[0]

    if len(step_indices) == 0:
        return None, t  # No valid step response found

    step_responses = []

    for idx in step_indices:
        start = max(0, idx)
        end = min(len(GY), start + segment_length)

        if end - start < len(t):  # Ensure enough data for response
            continue

        sp_segment = SP[start:end]
        gy_segment = GY[start:end]

        # Compute impulse response via deconvolution
        h, _ = signal.deconvolve(gy_segment, sp_segment)
        step_resp = np.cumsum(h[:len(t)])  # Integrate to get step response

        if Ycorrection:
            y_offset = 1 - np.mean(step_resp[int(len(t) * 0.4):])
            step_resp *= (y_offset + 1)

        step_responses.append(step_resp)

    if not step_responses:
        return None, t  # No valid responses

    # Average step responses
    step_response_avg = np.mean(step_responses, axis=0)

    return step_response_avg, t

def StepResponse(SP, GY, lograte = 1, Ycorrection = 1.0):

    minInput = 20
    segment_length = (lograte * 1000)  # 3 sec segments
    wnd = (lograte * 1000) * .5  # 500ms step response function, length will depend on lograte
    StepRespDuration_ms =  500  # max dur of step resp in ms for plotting
    t = np.arange(start=0, stop=StepRespDuration_ms + (1 / lograte), step=(1 / lograte))  # % time in ms
    fileDurSec = len(SP) / (lograte * 1000)

    subsampleFactor = 1;
    if fileDurSec <= 20:
        subsampleFactor = 10;
    elif fileDurSec > 20 and fileDurSec <= 60:
        subsampleFactor = 7;
    elif fileDurSec > 60:
        subsampleFactor = 3;

    stepresponse = [];
    segment_vector = np.arange(start=1, stop=len(SP), step=round(segment_length / subsampleFactor))
    segment_vector_added = segment_vector + int(segment_length)

    segment_last_element = segment_vector[len(segment_vector) - 1]

    for i in range(len(segment_vector_added)):
        if segment_vector_added[i] < segment_last_element:
            NSegs = i

    if NSegs > 0:
        SPseg = []  # empty matrix with 4000 columns
        GYseg = []
        resptmp = []
        j = 0
        for i in range(NSegs):
            index1 = segment_vector[i]
            index2 = segment_vector[i] + segment_length
            abs_max_value = max(map(abs, SP[index1: index2]))
            # abs_max_value = max(SP[segment_vector[i]: segment_vector[i] + segment_length], key=abs)
            if abs_max_value >= minInput:
                j = j + 1
                SPseg.append(SP[segment_vector[i]: segment_vector[i] + segment_length])
                GYseg.append(GY[segment_vector[i]: segment_vector[i] + segment_length])

        #print(np.shape(SPseg))
        #print(np.shape(GYseg))
        padLength = 100
        j = 0;
        #print(len(SPseg[0]))
        if SPseg:  # if array not empty
            for i in range(len(SPseg)):
                a = np.multiply(GYseg[i][:], np.hanning(len(GYseg[i])))
                b = np.multiply(SPseg[i][:], np.hanning(len(SPseg[i])))

                # a = fft([zeros(1, padLength) a zeros(1, padLength)]);
                a = np.fft.fft(np.concatenate([np.zeros(padLength), a, np.zeros(padLength)]))
                b = np.fft.fft(np.concatenate([np.zeros(padLength), b, np.zeros(padLength)]))

                G = a / len(a);
                H = b / len(b);
                Hcon = np.conj(H);

                imp = np.real(np.fft.ifft(
                    np.divide(np.multiply(G, Hcon), np.multiply(H, Hcon + 0.0001))))  # impulse response function
                resptmp.append(np.cumsum(imp))  # integrate impulse resp function

                # steadyStateWindow = find(t > 200 & t < StepRespDuration_ms);
                steadyStateWindow = np.where((t > 200) & (t < StepRespDuration_ms))
                steadyStateResp = resptmp[i][steadyStateWindow]

                if Ycorrection:
                    if np.mean(steadyStateResp) < 1 or np.mean(steadyStateResp) > 1:
                        yoffset = 1 - np.mean(steadyStateResp);
                        resptmp[i][:] = np.multiply(resptmp[i][:], (yoffset + 1))
                        abc = 0

                    steadyStateResp = resptmp[i][steadyStateWindow]

                # else:

                if np.min(steadyStateResp) > 0.5 and np.max(steadyStateResp) < 3:  # Quality control
                    j = j + 1
                    stepresponse.append(resptmp[i][0:int(1 + wnd)])

    return stepresponse, t

def plotData(x_axis_array, y_axis_array, plot_title: str, x_axis_title: str, y_axis_title: str, legend_title, plot_type: chr, ax, grid, ylim1=-1, ylim2=-1):

    plt.title(plot_title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    #plt.rc('axes', titlesize=8)  # fontsize of the axes title
    #plt.rc('axes', labelsize=10)  # fontsize of the x and y labels

    if ylim1 != -1 and ylim2 != -1:
        plt.ylim(ylim1, ylim2)  # limit y axis between var1 and var2

    if grid == 'major':
        ax.grid(which=grid)  # Add lines to grid
        ax.set_xticks(np.arange(0, x_axis_array[len(x_axis_array)-1], 20))
    elif grid == 'minor':
        ax.grid(which=grid, linestyle = '--')  # Add lines to grid
        ax.set_xticks(np.arange(0, x_axis_array[len(x_axis_array)-1], 2), minor = True)

    plt.plot(x_axis_array, y_axis_array, plot_type, label = legend_title)
    plt.legend(loc="upper right")


def plotFFTData(x_axis_array, y_axis_array, peak_array, plot_title: str, x_axis_title: str, y_axis_title: str, legend_title, plot_type: chr, ax, grid, ylim1=-1, ylim2=-1):

    plt.title(plot_title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)

    if ylim1 != -1 and ylim2 != -1:
        plt.ylim(ylim1, ylim2)  # limit y axis between var1 and var2

    if grid == 'major':
        ax.grid(which=grid)  # Add lines to grid
        ax.set_xticks(np.arange(0, x_axis_array[len(x_axis_array)-1], 200))  # put major lines at every 500th from 0 to 4000
    elif grid == 'minor':
        x = 0
        ax.grid(which=grid, linestyle = '--')  # Add lines to grid
        ax.set_xticks(np.arange(0, x_axis_array[len(x_axis_array)-1], 20), minor = True)  # put major lines at every 500th from 0 to 4000

    plt.plot(x_axis_array, y_axis_array, plot_type, label = legend_title)

    # Show at which frequencies there was a peak
    peaks_axis = (np.int16)(freq_raw[peak_array])

    # Put dots on the peak freq points
    for i in range(len(peaks_axis)):
        # plt.plot(peaks_x_axis[i], 0.2, plot_type)
        plt.plot(peaks_axis[i], y_axis_array[i], plot_type)

    plt.legend(loc="upper right")


# First sector is used to store information necessary to read file correctly
# such as how many sector are written etc.
def Read_InfoSector():
    currentBytes_binary = logFile.read(SECTOR_SIZE)

    print(currentBytes_binary)

    # ord converts ascii to integer
    if (currentBytes_binary[0] == ord('C') and currentBytes_binary[1] == ord('K')):

        # Firmware Details
        firmware_detail = currentBytes_binary[0:12]

        # Main Loop Time
        main_loop_time = (np.uint16)(currentBytes_binary[12] << 8 | currentBytes_binary[13])

        # Gyro Signs
        current1 = currentBytes_binary[14]
        current2 = currentBytes_binary[15]
        current3 = currentBytes_binary[16]

        gyro_sign = []
        if(current1 == 1):
            gyro_sign.append(-1)
        else:
            gyro_sign.append(1)

        if (current2 == 1):
            gyro_sign.append(-1)
        else:
            gyro_sign.append(1)

        if (current3 == 1):
            gyro_sign.append(-1)
        else:
            gyro_sign.append(1)

        # Gyro Scale
        gyro_scale = (float)(currentBytes_binary[17] << 8 | currentBytes_binary[18])
        gyro_scale /= 1e5

        # Bytes Per Log
        bytes_per_log = (np.uint8)(currentBytes_binary[19])

        # Filter Settings

        # Gyro LPF cutoff
        use_lpf = (np.uint8)(currentBytes_binary[20])
        lpf_cutoff = (np.uint8)(currentBytes_binary[21])

        # Notch 1
        use_notch1 = (np.uint8)(currentBytes_binary[22])
        notch1_min = (np.uint16)(currentBytes_binary[23] << 8 | currentBytes_binary[24])
        notch1_max = (np.uint16)(currentBytes_binary[25] << 8 | currentBytes_binary[26])

        # Notch 2
        use_notch2 = (np.uint8)(currentBytes_binary[27])
        notch2_min = (np.uint16)(currentBytes_binary[28] << 8 | currentBytes_binary[29])
        notch2_max = (np.uint16)(currentBytes_binary[30] << 8 | currentBytes_binary[31])

        # Notch 3
        use_notch3 = (np.uint8)(currentBytes_binary[32])
        notch3_min = (np.uint16)(currentBytes_binary[33] << 8 | currentBytes_binary[34])
        notch3_max = (np.uint16)(currentBytes_binary[35] << 8 | currentBytes_binary[36])

        # PID Parameters
        pid_parameters = np.zeros((5, 3))
        pid_parameters[0][0] = currentBytes_binary[37]
        pid_parameters[0][1] = currentBytes_binary[38]
        pid_parameters[0][2] = currentBytes_binary[39]
        pid_parameters[3][0] = currentBytes_binary[40]
        pid_parameters[4][0] = currentBytes_binary[41]

        pid_parameters[1][0] = currentBytes_binary[42]
        pid_parameters[1][1] = currentBytes_binary[43]
        pid_parameters[1][2] = currentBytes_binary[44]
        pid_parameters[3][1] = currentBytes_binary[45]
        pid_parameters[4][1] = currentBytes_binary[46]

        pid_parameters[2][0] = currentBytes_binary[47]
        pid_parameters[2][1] = currentBytes_binary[48]
        pid_parameters[2][2] = currentBytes_binary[49]
        pid_parameters[3][2] = currentBytes_binary[50]
        pid_parameters[4][2] = currentBytes_binary[51]

        pid_master_multiplier   = float(currentBytes_binary[52] / 100.0)
        pid_pi_gain             = float(currentBytes_binary[53] / 100.0)
        pid_ff_gain             = float(currentBytes_binary[54] / 100.0)
        pid_roll_pitch_ratio    = float(currentBytes_binary[55] / 100.0)
        pid_i_gain              = float(currentBytes_binary[56] / 100.0)
        pid_d_gain              = float(currentBytes_binary[57] / 100.0)
        pid_d_max_gain          = float(currentBytes_binary[58] / 100.0)
        pid_pitch_pi_gain       = float(currentBytes_binary[59] / 100.0)

        # Log ok text
        log_ok = currentBytes_binary[194:200]

        # Total sectors written
        current1 = currentBytes_binary[200]
        current2 = currentBytes_binary[201]
        current3 = currentBytes_binary[202]
        current4 = currentBytes_binary[203]

        total_sectors = (np.uint32)((current1 << 24) | (current2 << 16) | (current3 << 8) | (current4))

        # Total sectors written
        current1 = currentBytes_binary[204]
        current2 = currentBytes_binary[205]
        current3 = currentBytes_binary[206]
        current4 = currentBytes_binary[207]

        flight_log_time = (np.uint32)((current1 << 24) | (current2 << 16) | (current3 << 8) | (current4))

        # Invalid data
        current1 = currentBytes_binary[208]
        current2 = currentBytes_binary[209]
        current3 = currentBytes_binary[210]
        current4 = currentBytes_binary[211]

        invalid_data_counter = (np.uint32)((current1 << 24) | (current2 << 16) | (current3 << 8) | (current4))

        return total_sectors, log_ok, main_loop_time, gyro_sign, gyro_scale, bytes_per_log, firmware_detail, +\
               use_lpf, lpf_cutoff, use_notch1, notch1_min, notch1_max, +\
               use_notch2, notch2_min, notch2_max, use_notch3, notch3_min, notch3_max, flight_log_time, +\
               pid_parameters, pid_master_multiplier, pid_pi_gain, pid_ff_gain, pid_roll_pitch_ratio, pid_i_gain, +\
               pid_d_gain, pid_d_max_gain, pid_pitch_pi_gain, invalid_data_counter


# 'rb' reads data in binary mode meaning that the data is not ascii encoded. each data is regarded and used as binary value
# when 'r' used with encoding such as "ISO-8859-1" then for example 0x13 is not readable etc.
# flight controller logs the data without encoding.

firmware_detail = []
# empty arrays to append new bytes to end.
gyro_raw_x = []
gyro_preLPF_x = []
gyro_preNotch_x = []
gyro_filtered_x = []
gyro_zero_x = []

gyro_raw_y = []
gyro_preLPF_y = []
gyro_preNotch_y = []
gyro_filtered_y = []
gyro_zero_y = []

gyro_raw_z = []
gyro_preLPF_z = []
gyro_preNotch_z = []
gyro_filtered_z = []
gyro_zero_z = []

altHold_throttle = []

rc_data_roll = []
rc_setpoint_roll = []

rc_data_pitch = []
rc_setpoint_pitch = []

rc_data_yaw = []
rc_setpoint_yaw = []

rc_data_throttle = []

motor_final1 = []
motor_final2 = []
motor_final3 = []
motor_final4 = []

imu_x = []
imu_y = []
imu_z = []

loop_cycle = []
loop_time = []
flags = []

acc_filtered_x = []
acc_filtered_y = []
acc_filtered_z = []

pid_p_x = []
pid_i_x = []
pid_d_x = []
pid_f_x = []
pid_x = []

pid_p_y = []
pid_i_y = []
pid_d_y = []
pid_f_y = []
pid_y = []

pid_p_z = []
pid_i_z = []
pid_d_z = []
pid_f_z = []
pid_z = []
timeArray = []

SECTOR_SIZE = 512

START_INDICATOR = ord('C')
END_INDICATOR = ord('K')

logFile = open("/Users/ck/Desktop/flight_log.txt", "rb")
#logFile = open("flight_log.txt", "rb")

currentTime     = 0
sector_counter  = 0
print_sector    = 0
write_step_data = 0
check_step_resp = 1

(TOTAL_SECTORS, LOG_OK, MAIN_LOOP_TIME, GYRO_SIGN, GYRO_SCALE, BYTES_PER_LOG, FIRMWARE_DETAILS,
 USE_LPF, LPF_CUTOFF, USE_NOTCH1, NOTCH1_MIN, NOTCH1_MAX, USE_NOTCH2, NOTCH2_MIN, NOTCH2_MAX,
 USE_NOTCH3, NOTCH3_MIN, NOTCH3_MAX, FLIGHT_LOG_TIME, PID_PARAMETERS,
 PID_MASTER_MULTIPLIER, PID_PI_GAIN, PID_FF_GAIN, PID_ROLL_PITCH_RATIO, PID_I_GAIN,
 PID_D_GAIN, PDI_D_MAX_GAIN, PDI_PITCH_PI_GAIN, INVALID_DATA_COUNTER) = Read_InfoSector()

SAMPLE_PERIOD = (MAIN_LOOP_TIME * 0.000001)
SAMPLE_FREQUENCY = 1 / SAMPLE_PERIOD
TIME_INCREMENT = SAMPLE_PERIOD
LOG_RATE = ((512 / BYTES_PER_LOG) * TOTAL_SECTORS) / FLIGHT_LOG_TIME
if LOG_RATE < 1:
    LOG_RATE = 1

print(PID_PARAMETERS)

print("Started Reading")
print("FIRMWARE: ", FIRMWARE_DETAILS)
print("LOG STAT: ", LOG_OK)
print("Sample Period:", SAMPLE_PERIOD)
print("Sample Freq:", SAMPLE_FREQUENCY)
print("Flight Log Time:", FLIGHT_LOG_TIME / 1000.0)
print("Total Sectors:", TOTAL_SECTORS)
print("Log Rate KHz:", LOG_RATE)
print("Inc Time:", TIME_INCREMENT)
print("Gyro Scale:", GYRO_SCALE)
print("Gyro Sign:", GYRO_SIGN[0], " ", GYRO_SIGN[1], " ", GYRO_SIGN[2])
print("Bytes Per Log:", BYTES_PER_LOG)
print("Invalid Data Number:", INVALID_DATA_COUNTER)


while (True):

    # current sector
    sector_counter += 1
    if sector_counter % 1000 == 0:
        print("Curren Sector: ", sector_counter)

    if (sector_counter == TOTAL_SECTORS):
        sector_counter = 0
        break  # finish while loop

    currentBytes_binary = logFile.read(SECTOR_SIZE)
    curr_len = len(currentBytes_binary)

    # Print the content of a sector if enabled
    if (print_sector):
        for i in range(16):
            for c in range(32):
                print(currentBytes_binary[i * 32 + c], end=" ")
            print(' ')

    # current byte of the sector
    byte_index = np.uint16(0) # it was making it 8 byte i started as 16 bit
    while (byte_index < SECTOR_SIZE):

        res = currentBytes_binary[byte_index]

        # if 1st is start byte start decoding
        if (res == START_INDICATOR):

            res = currentBytes_binary[int(byte_index + (BYTES_PER_LOG - 1))]

            # if BYTES_PER_LOG is end byte start decoding
            if (res == END_INDICATOR):

                # Gyro raw and filtered results.
                for axis in range(3):

                    byte_index += 1
                    gyroRaw_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    gyroRaw_LSB = np.uint8(currentBytes_binary[byte_index])

                    gyroRaw = np.int16(gyroRaw_MSB << 8 | gyroRaw_LSB)
                    gyroRaw *= GYRO_SCALE

                    byte_index += 1
                    gyroFiltered_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    gyroFiltered_LSB = np.uint8(currentBytes_binary[byte_index])

                    gyroFiltered = np.int16(gyroFiltered_MSB << 8 | gyroFiltered_LSB)

                    byte_index += 1
                    gyroPreLPF_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    gyroPreLPF_LSB = np.uint8(currentBytes_binary[byte_index])

                    gyroPreLPF = np.int16(gyroPreLPF_MSB << 8 | gyroPreLPF_LSB)

                    byte_index += 1
                    gyroPreNotch_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    gyroPreNotch_LSB = np.uint8(currentBytes_binary[byte_index])

                    gyroPreNotch = np.int16(gyroPreNotch_MSB << 8 | gyroPreNotch_LSB)

                    byte_index += 1
                    gyroZero_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    gyroZero_LSB = np.uint8(currentBytes_binary[byte_index])

                    gyroZero = np.int16(gyroZero_MSB << 8 | gyroZero_LSB)

                    # add new data to arrays
                    if (axis == 0):
                        gyro_raw_x.append(gyroRaw)
                        gyro_filtered_x.append(gyroFiltered)
                        gyro_preLPF_x.append(gyroPreLPF)
                        gyro_preNotch_x.append(gyroPreNotch)
                        gyro_zero_x.append(gyroZero)
                    elif (axis == 1):
                        gyro_raw_y.append(gyroRaw)
                        gyro_filtered_y.append(gyroFiltered)
                        gyro_preLPF_y.append(gyroPreLPF)
                        gyro_preNotch_y.append(gyroPreNotch)
                        gyro_zero_y.append(gyroZero)
                    elif (axis == 2):
                        gyro_raw_z.append(gyroRaw)
                        gyro_filtered_z.append(gyroFiltered)
                        gyro_preLPF_z.append(gyroPreLPF)
                        gyro_preNotch_z.append(gyroPreNotch)
                        gyro_zero_z.append(gyroZero)

                # Altitude hold adjustment result
                byte_index += 1
                altHold_thr = np.uint8(currentBytes_binary[byte_index])
                altHold_throttle.append(altHold_thr * 10)

                # Roll, pitch, yaw, throttle rc data
                for axis in range(4):

                    byte_index += 1
                    rcData_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    rcData_LSB = np.uint8(currentBytes_binary[byte_index])

                    rcData = np.uint16(rcData_MSB << 8 | rcData_LSB)

                    # No rc_setpoint for throttle
                    if axis != 3:
                        byte_index += 1
                        rcSetpoint_MSB = np.uint8(currentBytes_binary[byte_index])
                        byte_index += 1
                        rcSetpoint_LSB = np.uint8(currentBytes_binary[byte_index])

                        rcSetpoint = np.int16(rcSetpoint_MSB << 8 | rcSetpoint_LSB)

                    if (axis == 0):
                        rc_data_roll.append(rcData)
                        rc_setpoint_roll.append(rcSetpoint)
                    elif (axis == 1):
                        rc_data_pitch.append(rcData)
                        rc_setpoint_pitch.append(rcSetpoint)
                    elif (axis == 2):
                        rc_data_yaw.append(rcData)
                        rc_setpoint_yaw.append(rcSetpoint)
                    elif (axis == 3):
                        rc_data_throttle.append(rcData)
                        # No throttle in rc_setpoint

                # Motor final results
                for esc in range(4):
                    byte_index += 1
                    motorfinal = (np.uint8)(currentBytes_binary[byte_index])

                    if (esc == 0):
                        motor_final1.append(motorfinal * 10)
                    elif (esc == 1):
                        motor_final2.append(motorfinal * 10)
                    elif (esc == 2):
                        motor_final3.append(motorfinal * 10)
                    elif (esc == 3):
                        motor_final4.append(motorfinal * 10)

                # IMU angles
                for axis in range(3):
                    byte_index += 1
                    imu = np.uint8(currentBytes_binary[byte_index])

                    if (axis == 0):
                        imu_x.append(imu)
                    elif (axis == 1):
                        imu_y.append(imu)
                    elif (axis == 2):
                        imu_z.append(imu * 2)

                # Loop cycle time
                byte_index += 1
                loopCycle_MSB = np.uint8(currentBytes_binary[byte_index])
                byte_index += 1
                loopCycle_LSB = np.uint8(currentBytes_binary[byte_index])
                loopCycle = np.int16(loopCycle_MSB << 8 | loopCycle_LSB)
                loop_cycle.append(loopCycle)
                #loop_time.append()

                # 2 bytes for flags later decode it
                byte_index += 1
                flags_MSB = np.uint8(currentBytes_binary[byte_index])
                byte_index += 1
                flags_LSB = np.uint8(currentBytes_binary[byte_index])
                flags_bits = np.int16(flags_MSB << 8 | flags_LSB)
                flags.append(flags_bits)

                # Acc filtered results.
                for axis in range(3):

                    byte_index += 1
                    accFiltered_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    accFiltered_LSB = np.uint8(currentBytes_binary[byte_index])

                    accFiltered = np.int16(accFiltered_MSB << 8 | accFiltered_LSB)

                    if (axis == 0):
                        acc_filtered_x.append(accFiltered)  # add new data to array
                    elif (axis == 1):
                        acc_filtered_y.append(accFiltered)
                    elif (axis == 2):
                        acc_filtered_z.append(accFiltered)

                # PID results.
                for axis in range(3):

                    byte_index += 1
                    pid_p_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    pid_p_LSB = np.uint8(currentBytes_binary[byte_index])

                    pid_p = np.int16(pid_p_MSB << 8 | pid_p_LSB)

                    byte_index += 1
                    pid_i_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    pid_i_LSB = np.uint8(currentBytes_binary[byte_index])

                    pid_i = np.int16(pid_i_MSB << 8 | pid_i_LSB)

                    byte_index += 1
                    pid_d_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    pid_d_LSB = np.uint8(currentBytes_binary[byte_index])

                    pid_d = np.int16(pid_d_MSB << 8 | pid_d_LSB)

                    byte_index += 1
                    pid_f_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    pid_f_LSB = np.uint8(currentBytes_binary[byte_index])

                    pid_f = np.int16(pid_f_MSB << 8 | pid_f_LSB)

                    byte_index += 1
                    pid_MSB = np.uint8(currentBytes_binary[byte_index])
                    byte_index += 1
                    pid_LSB = np.uint8(currentBytes_binary[byte_index])

                    pid = np.int16(pid_MSB << 8 | pid_LSB)

                    if (axis == 0):
                        pid_p_x.append(float(pid_p / 10.0))
                        pid_i_x.append(float(pid_i / 10.0))
                        pid_d_x.append(float(pid_d / 10.0))
                        pid_f_x.append(float(pid_f / 10.0))
                        pid_x.append(float(pid / 10.0))
                    elif (axis == 1):
                        pid_p_y.append(float(pid_p / 10.0))
                        pid_i_y.append(float(pid_i / 10.0))
                        pid_d_y.append(float(pid_d / 10.0))
                        pid_f_y.append(float(pid_f / 10.0))
                        pid_y.append(float(pid / 10.0))
                    elif (axis == 2):
                        pid_p_z.append(float(pid_p / 10.0))
                        pid_i_z.append(float(pid_i / 10.0))
                        pid_d_z.append(float(pid_d / 10.0))
                        pid_f_z.append(float(pid_f / 10.0))
                        pid_z.append(float(pid / 10.0))

                # 34 bytes available for later logging and last byte is end byte
                byte_index += 34

                # end byte is checked at the beginning
                byte_index += 1

                currentTime += TIME_INCREMENT
                timeArray.append(currentTime)

        else:
            byte_index += 1

print("End of Reading")

print("FIRMWARE: ", FIRMWARE_DETAILS)
print("LOG STAT: ", LOG_OK)
print("Sample Period:", SAMPLE_PERIOD)
print("Sample Freq:", SAMPLE_FREQUENCY)
print("Flight Log Time:", FLIGHT_LOG_TIME / 1000.0)
print("Total Sectors:", TOTAL_SECTORS)
print("Log Rate KHz:", LOG_RATE)
print("Inc Time:", TIME_INCREMENT)
print("Gyro Scale:", GYRO_SCALE)
print("Gyro Zero:", gyro_zero_x[0], " ", gyro_zero_y[1], " ", gyro_zero_z[2])
print("Gyro Sign:", GYRO_SIGN[0], " ", GYRO_SIGN[1], " ", GYRO_SIGN[2])
print("Bytes Per Log:", BYTES_PER_LOG)
print("Invalid Data Number:", INVALID_DATA_COUNTER)

#print("pid ff x:", pid_f_x)

if write_step_data:
    gyroFiltered_x_file = open("/home/ck/Desktop/gyroFilteredLog.txt", "w")
    setPoint_x_file = open("/home/ck/Desktop/setPointLog.txt", "w")

    gyroFiltered_x_file.write(str(gyro_filtered_x))
    print("gyro array length:", len(gyro_filtered_x))
    setPoint_x_file.write(str(rc_setpoint_roll))
    print("setpoint array length:", len(rc_setpoint_roll))

print("Plot qt window started")
pw = plotWindow()
print("Plot qt window ended")
if check_step_resp:

    window_order = 7
    ymin = 0
    ymax = 1.75
    for i in range(3):

        if i == 0:
            # Plot all in one page so set this only once
            fig = plt.figure(figsize=(8, 6))

        if i == 0:
            pid_parameters_roll_text = "P:" + str(int(PID_PARAMETERS[0][0])) + " I:" + str(int(PID_PARAMETERS[0][1])) \
                                       + " D:" + str(int(PID_PARAMETERS[0][2])) + " Dmin:" + str(int(PID_PARAMETERS[3][0])) \
                                       + " FF:" + str(int(PID_PARAMETERS[4][0]))
            ax = fig.add_subplot(3, 1, 1)
            ax.text(320, ymax + 0.25,
                    " MM: " + str(PID_MASTER_MULTIPLIER) +
                    " PI: " + str(PID_PI_GAIN) +
                    " PR: " + str(PID_ROLL_PITCH_RATIO) +
                    " FF: " + str(PID_FF_GAIN) +
                    " IG: " + str(PID_I_GAIN) +
                    " DG: " + str(PID_D_GAIN) +
                    " DMAX: " + str(PDI_D_MAX_GAIN) +
                    " PPI: " + str(PDI_PITCH_PI_GAIN))

            step_response_x, step_t_x = StepResponse(rc_setpoint_roll, gyro_filtered_x, LOG_RATE)
            mean_step_x = np.mean(step_response_x, axis=0)
            mean_step_x = savitzky_golay(mean_step_x, window_order, 3)
            plotData(step_t_x, mean_step_x, "Step Response Roll", "", "", pid_parameters_roll_text, 'lime', ax, 'major',ylim1=ymin, ylim2=ymax)
            #print(np.shape(step_t_x))
            print(np.shape(step_response_x))
            print(np.shape(mean_step_x))
            #print(mean_step_x)
        if i == 1:
            pid_parameters_pitch_text = "P:" + str(int(PID_PARAMETERS[1][0])) + " I:" + str(int(PID_PARAMETERS[1][1])) \
                                       + " D:" + str(int(PID_PARAMETERS[1][2])) + " Dmin:" + str(int(PID_PARAMETERS[3][1])) \
                                       + " FF:" + str(int(PID_PARAMETERS[4][1]))
            ax = fig.add_subplot(3, 1, 2)
            step_response_y, step_t_y = StepResponse(rc_setpoint_pitch, gyro_filtered_y, LOG_RATE)
            mean_step_y = np.mean(step_response_y, axis=0)
            mean_step_y = savitzky_golay(mean_step_y, window_order, 3)
            plotData(step_t_y, mean_step_y, "Step Response Pitch", "", "", pid_parameters_pitch_text, 'lime', ax, 'major',ylim1=ymin, ylim2=ymax)
            #print(np.shape(step_t_y))
            print(np.shape(step_response_y))
            print(np.shape(mean_step_y))
            #print(mean_step_y)
        if i == 2:
            pid_parameters_yaw_text = "P:" + str(int(PID_PARAMETERS[2][0])) + " I:" + str(int(PID_PARAMETERS[2][1])) \
                                        + " D:" + str(int(PID_PARAMETERS[2][2])) + " Dmin:" + str(int(PID_PARAMETERS[3][2])) \
                                       + " FF:" + str(int(PID_PARAMETERS[4][2]))
            ax = fig.add_subplot(3, 1, 3)
            step_response_z, step_t_z = StepResponse(rc_setpoint_yaw, gyro_filtered_z, LOG_RATE)
            mean_step_z = np.mean(step_response_z, axis=0)
            mean_step_z = savitzky_golay(mean_step_z, window_order, 3)
            plotData(step_t_z, mean_step_z, "Step Response Yaw", "", "", pid_parameters_yaw_text, 'lime', ax, 'major',ylim1=ymin, ylim2=ymax)
            #print(np.shape(step_t_z))
            print(np.shape(step_response_z))
            print(np.shape(mean_step_z))
            #print(mean_step_z)

    pw.addPlot("Step Response", fig)

for i in range(3):

    # Gyro Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 1, 1)

    # !!!!!Get gyro offset and subtract from raw and then multiply with sign
    if i == 0:
        gyro_raw = np.subtract(gyro_raw_x,gyro_zero_x)
        gyro_raw = [x * GYRO_SIGN[i] for x in gyro_raw]
        gyro_preLPF = gyro_preLPF_x
        gyro_preNotch = gyro_preNotch_x
        gyro_filtered = gyro_filtered_x
        rc_command = [x - 1500 for x in rc_data_roll] # subtract 1500 from each value to get command
        rc_setpoint = rc_setpoint_roll
        title = "Gyroscope X"
    elif i == 1:
        gyro_raw = np.subtract(gyro_raw_y,gyro_zero_y)
        gyro_raw = [x * GYRO_SIGN[i] for x in gyro_raw]
        gyro_preLPF = gyro_preLPF_y
        gyro_preNotch = gyro_preNotch_y
        gyro_filtered = gyro_filtered_y
        rc_command = [x - 1500 for x in rc_data_pitch]
        rc_setpoint = rc_setpoint_pitch
        title = "Gyroscope Y"
    elif i == 2:
        gyro_raw = np.subtract(gyro_raw_z, gyro_zero_z)
        gyro_raw = [x * GYRO_SIGN[i] for x in gyro_raw]
        gyro_preLPF = gyro_preLPF_z
        gyro_preNotch = gyro_preNotch_z
        gyro_filtered = gyro_filtered_z
        rc_command = [x - 1500 for x in rc_data_yaw]
        rc_setpoint = rc_setpoint_yaw
        title = "Gyroscope Z"

    window_order = 7
    # This one works great plots a perfect smooth line fitting to data without changing the array size
    rc_command_smooth = savitzky_golay(rc_command, window_order, 3)
    rc_setpoint_smooth = savitzky_golay(rc_setpoint, window_order, 3)

    # raw is data read from gyro without offset subraction
    # preNotch is (raw - offset) * scale data
    # preLPF is preNotch + notch applied gyro data
    # gyro_filtered is preNotch + notch + lpf applied gyro data
    gyro_raw_smooth = savitzky_golay(gyro_raw, window_order, 3)
    gyro_preNotch_smooth = savitzky_golay(gyro_preNotch, window_order, 3)
    gyro_preLPF_smooth = savitzky_golay(gyro_preLPF, window_order, 3)
    gyro_filtered_smooth = savitzky_golay(gyro_filtered, window_order, 3)

    # Plot to related tab
    plotData(timeArray, rc_command_smooth, title, "", "", "rc_command", 'blue', ax, 'minor')
    plotData(timeArray, rc_setpoint_smooth, title, "", "", "rc_setpoint", 'red', ax, 'minor')

    plotData(timeArray, gyro_raw_smooth, "", "", "", "gyro_raw", 'black', ax, 'minor')
    plotData(timeArray, gyro_preNotch_smooth, "", "", "", "(gyro_raw-offset)*scaled", 'magenta', ax, 'minor')
    plotData(timeArray, gyro_preLPF_smooth, "", "", "", "gyro+notch", 'cyan', ax, 'minor')
    plotData(timeArray, gyro_filtered_smooth, title, "Time (sec)", "Deg/sec", "gyro+notch+lpf", 'lime', ax, 'major')

    # Calculate FFT
    freq_raw, fft_raw, peaksFFT_raw = FFT_Calculate(gyro_raw, SAMPLE_PERIOD)
    freq_preLPF, fft_preLPF, peaksFFT_preLPF = FFT_Calculate(gyro_preLPF, SAMPLE_PERIOD)
    freq_preNotch, fft_preNotch, peaksFFT_preNotch = FFT_Calculate(gyro_preNotch, SAMPLE_PERIOD)
    freq_filtered, fft_filtered, peaksFFT_filtered = FFT_Calculate(gyro_filtered, SAMPLE_PERIOD)

    # FFT Plot
    ax = fig.add_subplot(2, 1, 2)

    plotFFTData(freq_raw, fft_raw, peaksFFT_raw, "", "", "", "fft_raw", 'r+', ax, 'minor', 0, 2)
    plotFFTData(freq_preLPF, fft_preLPF, peaksFFT_preLPF, "", "", "", "fft_preLPF", 'deepskyblue', ax, 'minor', 0, 2)
    plotFFTData(freq_preNotch, fft_preNotch, peaksFFT_preNotch, "", "", "", "fft_preNotch", 'slateblue', ax, 'minor', 0, 2)
    plotFFTData(freq_filtered, fft_filtered, peaksFFT_filtered, "", "Frequency", "Amplitude", "fft_filtered", 'lime', ax, 'minor', 0, 2)
    # plt.text(2000, 0.10, (np.int16)(freq_filtered[peaksFFT_Filtered]), bbox=dict(facecolor='cyan', alpha=0.5))

    # Plot filter cutoffs as one vertical line for each
    if USE_LPF:
        plt.axvline(x=LPF_CUTOFF, color='r', label='axvline - full height')
    if USE_NOTCH1:
        plt.axvline(x=NOTCH1_MIN, color='b', label='axvline - full height')
        plt.axvline(x=NOTCH1_MAX, color='b', label='axvline - full height')
    if USE_NOTCH2:
        plt.axvline(x=NOTCH2_MIN, color='r', label='axvline - full height')
        plt.axvline(x=NOTCH2_MAX, color='r', label='axvline - full height')
    if USE_NOTCH3:
        plt.axvline(x=NOTCH3_MIN, color='b', label='axvline - full height')
        plt.axvline(x=NOTCH3_MAX, color='b', label='axvline - full height')

    pw.addPlot(title, fig) # add figure as a tab

# Motor Plot
for i in range(0, 4):

    if i == 0:
        # Plot all in one page so set this only once
        fig = plt.figure(figsize=(8, 6))

    if i == 0:
        title = "RC Roll"
        ax = fig.add_subplot(5, 1, 1)
        rc_command = [x - 1500 for x in rc_data_roll] # subtract 1500 from each value to get command
        rc_setpoint = rc_setpoint_roll
        color1 = 'lightskyblue'
        color2 = 'blueviolet'
        legend_info1 = "rc_command_roll"
        legend_info2 = "rc_setpoint_roll"
        plotData(timeArray, rc_command, title, "", "", legend_info1, color1, ax, 'minor')
        plotData(timeArray, rc_setpoint, title, "", "", legend_info2, color2, ax, 'minor')
    elif i == 1:
        title = "RC Pitch"
        ax = fig.add_subplot(5, 1, 2)
        rc_command = [x - 1500 for x in rc_data_pitch]
        rc_setpoint = rc_setpoint_pitch
        color1 = 'c'
        color2 = 'g'
        legend_info1 = "rc_command_pitch"
        legend_info2 = "rc_setpoint_pitch"
        plotData(timeArray, rc_command, title, "", "", legend_info1, color1, ax, 'minor')
        plotData(timeArray, rc_setpoint, title, "", "", legend_info2, color2, ax, 'minor')
    elif i == 2:
        title = "RC Yaw"
        ax = fig.add_subplot(5, 1, 3)
        rc_command = [x - 1500 for x in rc_data_yaw]
        rc_setpoint = rc_setpoint_yaw
        color1 = 'springgreen'
        color2 = 'darkolivegreen'
        legend_info1 = "rc_command_yaw"
        legend_info2 = "rc_setpoint_yaw"
        plotData(timeArray, rc_command, title, "", "", legend_info1, color1, ax, 'minor')
        plotData(timeArray, rc_setpoint, title, "", "", legend_info2, color2, ax, 'minor')
    elif i == 3:
        title = "RC Throttle"
        ax = fig.add_subplot(5, 1, 4)
        rc_command = rc_data_throttle#[x - 1500 for x in rc_data_throttle]
        color1 = 'r'
        legend_info1 = "rc_command_throttle"
        plotData(timeArray, rc_command, title, "", "", legend_info1, color1, ax, 'minor')


    # Plot just once
    if i == 3:
        title = "ESC"
        ax = fig.add_subplot(5, 1, 5)

        plotData(timeArray, motor_final1, title, "", "", "motor1", 'springgreen', ax, 'minor')
        plotData(timeArray, motor_final2, title, "", "", "motor2", 'lightskyblue', ax, 'minor')
        plotData(timeArray, motor_final3, title, "", "", "motor3", 'slateblue', ax, 'minor')
        plotData(timeArray, motor_final4, title, "", "", "motor4", 'lightpink', ax, 'minor')


pw.addPlot("RC/ESC", fig) # add figure as a tab

# PID Plot
# PLot p, i, d, f, result for 3 axid
for i in range(0, 3):

    if i == 0:
        # Plot all in one page so set this only once
        fig = plt.figure(figsize=(8, 6))

    if i == 0:
        title = "PID Roll"
        ax = fig.add_subplot(3, 1, 1)
        pid_p = pid_p_x
        pid_i = pid_i_x
        pid_d = pid_d_x
        pid_f = pid_f_x
        pid   = pid_x

        legend_info1 = "p_term"
        legend_info2 = "i_term"
        legend_info3 = "d_term"
        legend_info4 = "f_term"
        legend_info5 = "pid"

    elif i == 1:
        title = "PID Pitch"
        ax = fig.add_subplot(3, 1, 2)
        pid_p = pid_p_y
        pid_i = pid_i_y
        pid_d = pid_d_y
        pid_f = pid_f_y
        pid = pid_y

        legend_info1 = "p_term"
        legend_info2 = "i_term"
        legend_info3 = "d_term"
        legend_info4 = "f_term"
        legend_info5 = "pid"
    elif i == 2:
        title = "PID Yaw"
        ax = fig.add_subplot(3, 1, 3)
        pid_p = pid_p_z
        pid_i = pid_i_z
        pid_d = pid_d_z
        pid_f = pid_f_z
        pid = pid_z

        legend_info1 = "p_term"
        legend_info2 = "i_term"
        legend_info3 = "d_term"
        legend_info4 = "f_term"
        legend_info5 = "pid"

    #plotData(timeArray, pid_p, title, "", "", legend_info1, 'springgreen', ax, 'minor')
    #plotData(timeArray, pid_i, title, "", "", legend_info2, 'darkolivegreen', ax, 'minor')
    #plotData(timeArray, pid_d, title, "", "", legend_info3, 'r', ax, 'minor')
    plotData(timeArray, pid_f, title, "", "", legend_info4, 'blueviolet', ax, 'minor')
    #plotData(timeArray, pid, title, "", "", legend_info5, 'hotpink', ax, 'minor')

pw.addPlot("PID", fig) # add figure as a tab


# Inertia Measurement Unit
for i in range(3):

    # IMU Plot
    if i == 0:
        # Plot all in one page so set this only once
        fig = plt.figure(figsize=(8, 6))

    if i == 0:
        data = imu_x
        title = "Inertia Measurement Unit X"
        # Actual range is +- 180. To see with margin i set as below
        ylim1 = -300
        ylim2 = 300
        color = 'lime'
    elif i == 1:
        data = imu_y
        title = "Inertia Measurement Unit Y"
        ylim1 = -300
        ylim2 = 300
        color = 'yellowgreen'
    elif i == 2:
        data = imu_z
        title = "Inertia Measurement Unit Z"
        # Actual range is 0 to 360. To see with margin i set as below
        ylim1 = 0
        ylim2 = 400
        color = 'y'

    ax = fig.add_subplot(3, 1, i+1)
    plotData(timeArray, data, title, "Time (Sec)", "Degrees", title, color, ax, 'minor', ylim1, ylim2)

pw.addPlot("Inertia Measurement Unit", fig)

# Accelerometer

for i in range(3):

    # ACC Plot
    if i == 0:
        # Plot all in one page so set this only once
        fig = plt.figure(figsize=(8, 6))

    if i == 0:
        data = acc_filtered_x
        title = "Accelerometer X"
        ylim1 = -3000
        ylim2 = 3000
        color = 'lime'

    elif i == 1:
        data = acc_filtered_y
        title = "Accelerometer Y"
        ylim1 = -3000
        ylim2 = 3000
        color = 'yellowgreen'

    elif i == 2:
        data = acc_filtered_z
        title = "Accelerometer Z"
        ylim1 = -100
        ylim2 = 5000
        color = 'y'

    ax = fig.add_subplot(3, 1, i+1)
    plotData(timeArray, data, title, "Time (Sec)", "LSB/g", title, color, ax, 'minor', ylim1, ylim2)

pw.addPlot("Accelerometer Fitered", fig)

# Loop Time
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

min_time = np.min(loop_cycle)
max_time = np.max(loop_cycle)
y_min = min_time - 25
y_max = max_time + 25

ax.text(0, y_max + 5, " Min Time: " + str(min_time) + " Max Time: " + str(max_time))

plotData(timeArray, loop_cycle, "Loop Time", "Time (Sec)", "", "Loop Time", 'springgreen', ax, 'minor', y_min, y_max)

pw.addPlot("Loop Time", fig)

pw.show()

# Add order to plots
#ax1.set_zorder(1)
# d and f is zero check it!!!
























