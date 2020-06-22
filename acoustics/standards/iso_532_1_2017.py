"""
ISO 532-1:2017
===============

ISO 532-1:2017 specifies two methods for estimating the loudness and loudness
level of sounds as perceived by otologically normal persons under specific
listening conditions. The first method is intended for stationary sounds and
the second method for arbitrary non-stationary (time-varying) sounds,
including stationary sounds as a special case.

----------------------------------------------

This code is based on the Matlab library created by Jaume Segura-Garcia,
Jesus Lopez-Ballester, and Adolfo Pastor-Aparicio:
https://github.com/jausegar/urbauramon/tree/master/URPAA/libraries/libNS

MATLAB equivalent variables:
Input variables:
LT(28)      Field of 28 elements which represent the 1/3 OB levels in dB with 
            fc = 25 Hz to 12.5 kHz
    replaced with OctaveBand object / np.array
    renamed: levels

MS          String variable to distinguish the type of sound field ( free / diffuse )
    renamed: field_type
-------------------------------
Output variables :
N           Loudness in sone G
NS          Specific Loudness
err         Error code

-------------------------------
Defined Variables:

FR(28)      Center frequencies of 1/3 OB
    substituted with: iec_61260_1_2014.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
    renamed: THIRD_OCTAVE_CENTER_FREQUENCIES

RAP(8)      Ranges of 1/3 OB levels for correction at low frequencies according to
            equal loudness contours.
    renamed: LOW_FREQ_CORRECTION_LEVEL_RANGES

DLL(11,8)   Reduction of 1/3 OB levels at low frequencies according to equal
            loudness contours within the 8 ranges defined by RAP
    renamed: THIRD_OCTAVE_LOW_FREQ_CORRECTIONS

LTQ(20)     Critical Band Rate level at absolute threshold without taking into
            account the transmission characteristics of the ear

AO(20)      Correction of levels according to the transmission characteristics
            of the ear
    renamed: A0

DDF(20)     Level difference between free and diffuse sound fields
    renamed: FREE_DIFFUSE_FIELDS_LEVEL_DIFFERENCE

DCB(20)     Adaptation of 1/3 OB levels to the corresponding critical band level

ZUP(21)     Upper limits of approximated critical bands in terms of critical band rate
    renamed: THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK

RNS(18)     Range of specific loudness for the determination of the steepness of the
            upper slopes in the specific loudness -critical band rate pattern
    renamed: SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES

USL(18,8)   Steepness of the upper slopes in the specific loudness - critical band rate
            pattern for the ranges RNS as a function of the number of the critical band
    renamed: CRITICAL_BAND_UPPER_SLOPE_STEEPNESS

-------------------------------------
Working variables (Uncertain of Definitions)

XP          Equal Loudness Contours
TI          Intensity of LT (levels)
    renamed: third_oct_intensity

LCB         Lower critical band
    renamed: lower_critical_band

LE          Level Excitation
    renamed: level_excitation

NM          Critical Band Level
    renamed: critical_band_level

KORRY       Correction Factor
    renamed: corr_factor

N           Loudness (in sones)
    renamed: loudness

DZ          Separation in CBR
    renamed:

N2          Main Loudness
    renamed:

Z1          Critical Band Rate for Lower Limit
    renamed:

N1          Loudness of previous band
    renamed:

IZ          Center "Frequency" Counter, used with NS
    renamed:

Z           Critical band rate
    renamed: critical_band_rate

J, IG       Counters used with USL
"""
import numpy as np
import acoustics
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)

#### Constants definitions

# Centre frequencies of 1/3 oct bands
THIRD_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[4:-2]

# Ranges of 1/3 Oct bands for correction at low frequencies according to
# equal loudness contours (Table A.3)
LOW_FREQ_CORRECTION_LEVEL_RANGES = np.array([45, 55, 65, 71, 80, 90, 100, 120])

# Reduction of 1/3 OB levels at low frequencies according to equal loudness
# contours within the 8 ranges defined by LOW_FREQ_CORRECTION_LEVEL_RANGES. 
# (Table A.3)
THIRD_OCTAVE_LOW_FREQ_CORRECTIONS = np.array(
    [
        [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
        [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
        [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
        [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
        [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
        [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
        [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
        [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0],
    ]
)

# Critical Band level at threshold in quiet (L_TQ, in dB) without 
# consideration of the ear's transmission characteristics. (Table A.6)
LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) # Threshold due to internal noise
# Hearing thresholds for the excitation levels (each number corresponds to a critical band 12.5 kHz is not included)

# Attenuation representing transmission between freefield and our hearing 
# system (Table A.4)
A0 = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6,
    -3.2, -5.4, -5.6, -4.0, -1.5, 2.0, 5.0, 12.0]
)
# Attenuation due to transmission in the middle ear
# Moore et al disagrees with this being flat for low frequencies

# Level correction to convert from a free field to a diffuse field 
# (Last critical band 12.5kHz is not included)
FREE_DIFFUSE_FIELDS_LEVEL_DIFFERENCE = np.array(
    [0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0,
    0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0, 4.0, 4.3, 4.0]
)

# Correction factor because using third octave band levels 
# (rather than critical bands) (Table A.7)
DCB = np.array(
    [-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5,
    1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0, -0.5]
)

# Mapping of centre freqs of 1/3 OB to Bark (critical band). Table A.8
THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK = np.array(
    [0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3,
    13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8, 22.7, 23.6, 24.0]
)

# Range of specific loudness for the determination of the steepness 
# of the upper slopes in the specific loudness - critical band 
# rate pattern. (Table A.9)
SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES = np.array(
    [21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13,
   1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10, 0.035, 0.0]
)

# Steepness of the upper slopes in the specific loudness - critical band rate
# pattern for the ranges SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES
#  as a function of the number of the critical band. (Table A.9)
CRITICAL_BAND_UPPER_SLOPE_STEEPNESS = np.array([
    [13.00, 8.20, 6.30, 5.50, 5.50, 5.50, 5.50, 5.50],
    [9.00, 7.50, 6.00, 5.10, 4.50, 4.50, 4.50, 4.50],
    [7.80, 6.70, 5.60, 4.90, 4.40, 3.90, 3.90, 3.90],
    [6.20, 5.40, 4.60, 4.00, 3.50, 3.20, 3.20, 3.20],
    [4.50, 3.80, 3.60, 3.20, 2.90, 2.70, 2.70, 2.70],
    [3.70, 3.00, 2.80, 2.35, 2.20, 2.20, 2.20, 2.20],
    [2.90, 2.30, 2.10, 1.90, 1.80, 1.70, 1.70, 1.70],
    [2.40, 1.70, 1.50, 1.35, 1.30, 1.30, 1.30, 1.30],
    [1.95, 1.45, 1.30, 1.15, 1.10, 1.10, 1.10, 1.10],
    [1.50, 1.20, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
    [0.40, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]]
    )



def stationary_loudness(levels, field_type):
    """Calculate loudness of a stationary signal

    Parameters
    ----------
    levels : np.array 
        Array containing third octave levels
    field_type : str 
        Variable to distinguish the type of sound field ( free / diffuse )

    Returns 
    ----------
    loudness : float
        Loudness in sone G
    specific_loudness : np.array
        Specific Loudness
    """

    ##### STEP 1: Weighting (Table A.3) #####
    # Correction of 1/3 OB levels according to equal loudness contours (XP) and
    # calculation of the intensities for 1/3 OB's up to 315 Hz

    third_oct_intensity = np.zeros(11)
    for i in range(11):
        j = 1
        while j < 7:
            if levels[i] <= LOW_FREQ_CORRECTION_LEVEL_RANGES[j] - THIRD_OCTAVE_LOW_FREQ_CORRECTIONS[j, i]:
                XP = levels[i] + THIRD_OCTAVE_LOW_FREQ_CORRECTIONS[j, i]
                third_oct_intensity[i] = 10**(0.1*XP)
                j = 8     # To exit from the while loop
        
            else:
                j = j + 1

    #### STEP 2 ####
    # Determination of Levels LCB(1), LCB(2), and LCB(3) within the first three
    # critical bands
    if np.count_nonzero(third_oct_intensity)-len(third_oct_intensity) != 0:
        print(' ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('WARNING!!!! ERROR!!!! One of the 1/3 OBs level exceeded the metrics range.')
        print('If you are using this code to normalize loudness, reduce the initial level and try again.')
        print('Otherwise any results if obtained will be incorrect.')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(' ')
        loudness = -1
        specific_loudness = -1 * np.ones(240)
        err = -1
        return loudness, specific_loudness, err

    lower_critical_band = np.zeros(3)
    lower_critical_band[0] = third_oct_intensity[0] +\
                             third_oct_intensity[1] +\
                             third_oct_intensity[2] +\
                             third_oct_intensity[3] +\
                             third_oct_intensity[4] +\
                             third_oct_intensity[5]
    lower_critical_band[1] = third_oct_intensity[6] + third_oct_intensity[7] + third_oct_intensity[8]
    lower_critical_band[2] = third_oct_intensity[9] + third_oct_intensity[10]
    lower_critical_band = 10*np.log10(lower_critical_band)

    #### Step 3: Core Loudness (Equation A.2) ####
    # Calculation of Main Loudness
    level_excitation = np.zeros(20)
    critical_band_level = np.zeros(21)
    for i in range(20):
        if i <= 2:
            level_excitation[i] = lower_critical_band[i]
        else:
            level_excitation[i] = levels[i + 8]
        level_excitation[i] = level_excitation[i] - A0[i]
        critical_band_level[i] = 0
        if field_type == 'diffuse':
            level_excitation[i] = level_excitation[i] + FREE_DIFFUSE_FIELDS_LEVEL_DIFFERENCE[i]


        if level_excitation[i] > LTQ[i]:
            level_excitation[i] = level_excitation[i] - DCB[i]
            # Core Loudness values, N_C (Equation A.2)
            s = 0.25    # threshold factor
            MP1 = 0.0635 * 10**(0.025*LTQ[i])
            MP2 = (1-s+s*10**(0.1*(level_excitation[i]-LTQ[i])))**0.25 - 1  # Equation A.2
            critical_band_level[i] = MP1 * MP2
            if critical_band_level[i] <= 0:
                critical_band_level[i] = 0

    critical_band_level[20] = 0     # unsure why this is set this way in the original Matlab code...

    # Correction of specific loudness in the lowest critical band taking
    # into account the dependence of absolute threshold within this critical band

    corr_factor = 0.4 + 0.32 * critical_band_level[0]**0.2
    if corr_factor > 1:
        corr_factor = 1
    critical_band_level[0] = critical_band_level[0] * corr_factor

    # Start Values
    loudness = 0
    Z1 = 0
    N1 = 0
    IZ = 0
    Z = 0.1
    short = 0
    specific_loudness = np.zeros(21)

    # Step to first and subsequent critical bands
    # TODO: Fix alllll of this
    for i in range(21):
        THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i] = THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i] + 0.0001
        IG = i-1
        if IG > 7:
            IG = 7

        while Z1 < THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i]:   # Note, Z1 will always be < ZUP[i] when line is first reached for each i
            if N1 > critical_band_level[i]:
                #TODO: Investigate this
                # Decide whether the critical band in question is completely or
                # partly masked by accessory loudness
                N2 = SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES[j] # NOTE: Not sure about this j used here
                if N2 < critical_band_level[i]:
                    N2 = critical_band_level[i]
                DZ = (N1 - N2) / CRITICAL_BAND_UPPER_SLOPE_STEEPNESS[j, IG]
                Z2 = Z1 + DZ
                if Z2 > THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i]:
                    Z2 = THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i]
                    DZ = Z2 - Z1
                    N2 = N1 - DZ * CRITICAL_BAND_UPPER_SLOPE_STEEPNESS[j, IG]

                # Contribution of accessory loudness to total loudness
                loudness = loudness + DZ * (N1 + N2) / 2
                while Z < Z2:
                    specific_loudness[IZ] = N1 - (Z - Z1) * CRITICAL_BAND_UPPER_SLOPE_STEEPNESS[j, IG]
                    IZ = IZ + 1
                    Z = Z + 0.1
            elif N1 == critical_band_level[i]:
                #TODO: Investigate this
                # Contribution of unmasked main loudness to total loudness and calculation
                # of values specific_loudness[IZ] with a spacing of Z = IZ * 0.1 Bark
                Z2 = THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i]
                N2 = critical_band_level[i]
                loudness = loudness + N2 * (Z2 - Z1)
                while Z < Z2:
                    specific_loudness[IZ] = N2
                    IZ = IZ + 1
                    Z = Z + 0.1

            else:
                # Determination of the number j corresponding to the range of specific loudness
                #XXX This block seems to be fine
                for j in range(18):
                    if SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES[j] < critical_band_level[i]:
                        break

                # Contribution of unmasked main loudness to total loudness and calculation
                # of values specific_loudness[IZ] with a spacing of Z = IZ*0.1 Bark
                Z2 = THIRD_OCTAVE_CENTER_FREQUENCIES_TO_BARK[i]
                N2 = critical_band_level[i]
                loudness = loudness + N2 * (Z2 - Z1)
                while Z < Z2:
                    specific_loudness[IZ] = N2
                    IZ = IZ + 1
                    Z = Z + 0.1
                    
        #XXX Breaks here
        while j < 18:
            # XXX Not looked into past here
            if N2 <= SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES[j]:
                j = j + 1
            else:
                break
        if N2 <= SPECIFIC_LOUDNESS_UPPER_SLOPE_STEEPNESS_RANGES[j] and j >= 18:
            j = 18

        Z1 = Z2
        N1 = N2

    # Now apply rounding types of corrections
    if loudness < 0:
        loudness = 0
    elif loudness <= 16:
        loudness = np.floor(loudness * 1000 + 0.5) / 1000
    else:
        loudness = np.floor(loudness * 100 + 0.5) / 100

    err = 0

    return loudness, specific_loudness, err


if __name__ == '__main__':
    s = acoustics.Signal.from_wav(
        'C:\\Users\Andrew\OneDrive - University College London\_PhD\Open Source Software\python-acoustics\data\ir_sportscentre_omni.wav')
    third_oct = s.third_octaves()
    levels = third_oct[1].leq()[4:-2]
    loudness, specific_loudness, err = stationary_loudness(levels, 'free')
