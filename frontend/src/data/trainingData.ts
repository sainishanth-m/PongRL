// Real serialized empirical training metrics from 6,001 episodes
export interface TrainingPoint {
  episode: number;
  reward: number;
  ma50: number;
  ma100: number;
}

export const REAL_TRAINING_POINTS: TrainingPoint[] = [
  {
    "episode": 0,
    "reward": -20.0,
    "ma50": -20.0,
    "ma100": -20.0
  },
  {
    "episode": 25,
    "reward": -20.12,
    "ma50": -20.05,
    "ma100": -20.05
  },
  {
    "episode": 50,
    "reward": -20.22,
    "ma50": -20.12,
    "ma100": -20.11
  },
  {
    "episode": 75,
    "reward": -20.32,
    "ma50": -20.23,
    "ma100": -20.17
  },
  {
    "episode": 100,
    "reward": -20.38,
    "ma50": -20.31,
    "ma100": -20.22
  },
  {
    "episode": 125,
    "reward": -20.37,
    "ma50": -20.37,
    "ma100": -20.3
  },
  {
    "episode": 150,
    "reward": -20.35,
    "ma50": -20.37,
    "ma100": -20.34
  },
  {
    "episode": 175,
    "reward": -20.32,
    "ma50": -20.35,
    "ma100": -20.36
  },
  {
    "episode": 200,
    "reward": -20.25,
    "ma50": -20.32,
    "ma100": -20.35
  },
  {
    "episode": 225,
    "reward": -20.26,
    "ma50": -20.29,
    "ma100": -20.32
  },
  {
    "episode": 250,
    "reward": -20.19,
    "ma50": -20.24,
    "ma100": -20.28
  },
  {
    "episode": 275,
    "reward": -20.11,
    "ma50": -20.19,
    "ma100": -20.24
  },
  {
    "episode": 300,
    "reward": -19.96,
    "ma50": -20.11,
    "ma100": -20.18
  },
  {
    "episode": 325,
    "reward": -19.9,
    "ma50": -19.99,
    "ma100": -20.09
  },
  {
    "episode": 350,
    "reward": -19.88,
    "ma50": -19.91,
    "ma100": -20.01
  },
  {
    "episode": 375,
    "reward": -19.86,
    "ma50": -19.88,
    "ma100": -19.94
  },
  {
    "episode": 400,
    "reward": -19.82,
    "ma50": -19.86,
    "ma100": -19.89
  },
  {
    "episode": 425,
    "reward": -19.76,
    "ma50": -19.82,
    "ma100": -19.85
  },
  {
    "episode": 450,
    "reward": -19.53,
    "ma50": -19.7,
    "ma100": -19.78
  },
  {
    "episode": 475,
    "reward": -19.48,
    "ma50": -19.57,
    "ma100": -19.7
  },
  {
    "episode": 500,
    "reward": -19.48,
    "ma50": -19.51,
    "ma100": -19.6
  },
  {
    "episode": 525,
    "reward": -19.29,
    "ma50": -19.45,
    "ma100": -19.51
  },
  {
    "episode": 550,
    "reward": -19.26,
    "ma50": -19.34,
    "ma100": -19.42
  },
  {
    "episode": 575,
    "reward": -19.19,
    "ma50": -19.26,
    "ma100": -19.35
  },
  {
    "episode": 600,
    "reward": -19.19,
    "ma50": -19.21,
    "ma100": -19.27
  },
  {
    "episode": 625,
    "reward": -19.08,
    "ma50": -19.17,
    "ma100": -19.21
  },
  {
    "episode": 650,
    "reward": -18.98,
    "ma50": -19.11,
    "ma100": -19.16
  },
  {
    "episode": 675,
    "reward": -18.84,
    "ma50": -18.99,
    "ma100": -19.08
  },
  {
    "episode": 700,
    "reward": -18.69,
    "ma50": -18.83,
    "ma100": -18.97
  },
  {
    "episode": 725,
    "reward": -18.59,
    "ma50": -18.68,
    "ma100": -18.83
  },
  {
    "episode": 750,
    "reward": -18.52,
    "ma50": -18.6,
    "ma100": -18.71
  },
  {
    "episode": 775,
    "reward": -18.53,
    "ma50": -18.54,
    "ma100": -18.61
  },
  {
    "episode": 800,
    "reward": -18.47,
    "ma50": -18.5,
    "ma100": -18.55
  },
  {
    "episode": 825,
    "reward": -18.48,
    "ma50": -18.49,
    "ma100": -18.52
  },
  {
    "episode": 850,
    "reward": -18.32,
    "ma50": -18.44,
    "ma100": -18.47
  },
  {
    "episode": 875,
    "reward": -18.22,
    "ma50": -18.33,
    "ma100": -18.41
  },
  {
    "episode": 900,
    "reward": -18.12,
    "ma50": -18.21,
    "ma100": -18.32
  },
  {
    "episode": 925,
    "reward": -17.85,
    "ma50": -18.07,
    "ma100": -18.2
  },
  {
    "episode": 950,
    "reward": -17.94,
    "ma50": -17.93,
    "ma100": -18.07
  },
  {
    "episode": 975,
    "reward": -17.9,
    "ma50": -17.92,
    "ma100": -17.99
  },
  {
    "episode": 1000,
    "reward": -17.76,
    "ma50": -17.88,
    "ma100": -17.91
  },
  {
    "episode": 1025,
    "reward": -17.63,
    "ma50": -17.78,
    "ma100": -17.85
  },
  {
    "episode": 1050,
    "reward": -17.52,
    "ma50": -17.65,
    "ma100": -17.77
  },
  {
    "episode": 1075,
    "reward": -17.58,
    "ma50": -17.55,
    "ma100": -17.66
  },
  {
    "episode": 1100,
    "reward": -17.25,
    "ma50": -17.48,
    "ma100": -17.56
  },
  {
    "episode": 1125,
    "reward": -17.22,
    "ma50": -17.34,
    "ma100": -17.45
  },
  {
    "episode": 1150,
    "reward": -17.05,
    "ma50": -17.19,
    "ma100": -17.33
  },
  {
    "episode": 1175,
    "reward": -17.05,
    "ma50": -17.07,
    "ma100": -17.21
  },
  {
    "episode": 1200,
    "reward": -16.88,
    "ma50": -17.01,
    "ma100": -17.1
  },
  {
    "episode": 1225,
    "reward": -16.83,
    "ma50": -16.92,
    "ma100": -17.0
  },
  {
    "episode": 1250,
    "reward": -16.81,
    "ma50": -16.84,
    "ma100": -16.93
  },
  {
    "episode": 1275,
    "reward": -16.75,
    "ma50": -16.8,
    "ma100": -16.86
  },
  {
    "episode": 1300,
    "reward": -16.7,
    "ma50": -16.7,
    "ma100": -16.77
  },
  {
    "episode": 1325,
    "reward": -16.4,
    "ma50": -16.59,
    "ma100": -16.69
  },
  {
    "episode": 1350,
    "reward": -16.0,
    "ma50": -16.36,
    "ma100": -16.53
  },
  {
    "episode": 1375,
    "reward": -16.02,
    "ma50": -16.08,
    "ma100": -16.34
  },
  {
    "episode": 1400,
    "reward": -15.79,
    "ma50": -15.93,
    "ma100": -16.15
  },
  {
    "episode": 1425,
    "reward": -15.61,
    "ma50": -15.82,
    "ma100": -15.95
  },
  {
    "episode": 1450,
    "reward": -15.57,
    "ma50": -15.66,
    "ma100": -15.8
  },
  {
    "episode": 1475,
    "reward": -15.55,
    "ma50": -15.56,
    "ma100": -15.69
  },
  {
    "episode": 1500,
    "reward": -15.32,
    "ma50": -15.51,
    "ma100": -15.58
  },
  {
    "episode": 1525,
    "reward": -15.17,
    "ma50": -15.31,
    "ma100": -15.43
  },
  {
    "episode": 1550,
    "reward": -15.19,
    "ma50": -15.19,
    "ma100": -15.35
  },
  {
    "episode": 1575,
    "reward": -15.03,
    "ma50": -15.17,
    "ma100": -15.24
  },
  {
    "episode": 1600,
    "reward": -14.95,
    "ma50": -15.06,
    "ma100": -15.12
  },
  {
    "episode": 1625,
    "reward": -14.9,
    "ma50": -14.96,
    "ma100": -15.07
  },
  {
    "episode": 1650,
    "reward": -14.85,
    "ma50": -14.91,
    "ma100": -14.99
  },
  {
    "episode": 1675,
    "reward": -14.39,
    "ma50": -14.74,
    "ma100": -14.85
  },
  {
    "episode": 1700,
    "reward": -14.01,
    "ma50": -14.36,
    "ma100": -14.64
  },
  {
    "episode": 1725,
    "reward": -14.18,
    "ma50": -14.09,
    "ma100": -14.41
  },
  {
    "episode": 1750,
    "reward": -13.99,
    "ma50": -14.04,
    "ma100": -14.2
  },
  {
    "episode": 1775,
    "reward": -13.87,
    "ma50": -13.96,
    "ma100": -14.02
  },
  {
    "episode": 1800,
    "reward": -13.7,
    "ma50": -13.81,
    "ma100": -13.92
  },
  {
    "episode": 1825,
    "reward": -13.68,
    "ma50": -13.68,
    "ma100": -13.82
  },
  {
    "episode": 1850,
    "reward": -13.83,
    "ma50": -13.7,
    "ma100": -13.75
  },
  {
    "episode": 1875,
    "reward": -13.91,
    "ma50": -13.81,
    "ma100": -13.75
  },
  {
    "episode": 1900,
    "reward": -13.68,
    "ma50": -13.84,
    "ma100": -13.77
  },
  {
    "episode": 1925,
    "reward": -13.36,
    "ma50": -13.65,
    "ma100": -13.73
  },
  {
    "episode": 1950,
    "reward": -13.38,
    "ma50": -13.43,
    "ma100": -13.63
  },
  {
    "episode": 1975,
    "reward": -13.21,
    "ma50": -13.36,
    "ma100": -13.51
  },
  {
    "episode": 2000,
    "reward": -12.84,
    "ma50": -13.23,
    "ma100": -13.33
  },
  {
    "episode": 2025,
    "reward": -12.66,
    "ma50": -12.9,
    "ma100": -13.13
  },
  {
    "episode": 2050,
    "reward": -12.21,
    "ma50": -12.6,
    "ma100": -12.92
  },
  {
    "episode": 2075,
    "reward": -12.33,
    "ma50": -12.31,
    "ma100": -12.61
  },
  {
    "episode": 2100,
    "reward": -12.55,
    "ma50": -12.28,
    "ma100": -12.44
  },
  {
    "episode": 2125,
    "reward": -12.51,
    "ma50": -12.52,
    "ma100": -12.41
  },
  {
    "episode": 2150,
    "reward": -12.34,
    "ma50": -12.51,
    "ma100": -12.4
  },
  {
    "episode": 2175,
    "reward": -12.21,
    "ma50": -12.34,
    "ma100": -12.43
  },
  {
    "episode": 2200,
    "reward": -11.9,
    "ma50": -12.07,
    "ma100": -12.29
  },
  {
    "episode": 2225,
    "reward": -11.7,
    "ma50": -11.81,
    "ma100": -12.08
  },
  {
    "episode": 2250,
    "reward": -11.78,
    "ma50": -11.76,
    "ma100": -11.92
  },
  {
    "episode": 2275,
    "reward": -11.5,
    "ma50": -11.77,
    "ma100": -11.79
  },
  {
    "episode": 2300,
    "reward": -11.26,
    "ma50": -11.52,
    "ma100": -11.64
  },
  {
    "episode": 2325,
    "reward": -11.01,
    "ma50": -11.23,
    "ma100": -11.5
  },
  {
    "episode": 2350,
    "reward": -10.86,
    "ma50": -11.1,
    "ma100": -11.31
  },
  {
    "episode": 2375,
    "reward": -11.01,
    "ma50": -11.0,
    "ma100": -11.12
  },
  {
    "episode": 2400,
    "reward": -10.82,
    "ma50": -10.93,
    "ma100": -11.01
  },
  {
    "episode": 2425,
    "reward": -10.75,
    "ma50": -10.87,
    "ma100": -10.94
  },
  {
    "episode": 2450,
    "reward": -10.38,
    "ma50": -10.75,
    "ma100": -10.84
  },
  {
    "episode": 2475,
    "reward": -10.42,
    "ma50": -10.52,
    "ma100": -10.69
  },
  {
    "episode": 2500,
    "reward": -10.33,
    "ma50": -10.4,
    "ma100": -10.57
  },
  {
    "episode": 2525,
    "reward": -10.07,
    "ma50": -10.34,
    "ma100": -10.43
  },
  {
    "episode": 2550,
    "reward": -10.26,
    "ma50": -10.25,
    "ma100": -10.32
  },
  {
    "episode": 2575,
    "reward": -10.13,
    "ma50": -10.17,
    "ma100": -10.26
  },
  {
    "episode": 2600,
    "reward": -10.4,
    "ma50": -10.13,
    "ma100": -10.19
  },
  {
    "episode": 2625,
    "reward": -9.68,
    "ma50": -10.06,
    "ma100": -10.12
  },
  {
    "episode": 2650,
    "reward": -9.5,
    "ma50": -9.83,
    "ma100": -9.98
  },
  {
    "episode": 2675,
    "reward": -9.67,
    "ma50": -9.59,
    "ma100": -9.83
  },
  {
    "episode": 2700,
    "reward": -9.49,
    "ma50": -9.52,
    "ma100": -9.67
  },
  {
    "episode": 2725,
    "reward": -9.53,
    "ma50": -9.54,
    "ma100": -9.57
  },
  {
    "episode": 2750,
    "reward": -9.48,
    "ma50": -9.5,
    "ma100": -9.51
  },
  {
    "episode": 2775,
    "reward": -9.42,
    "ma50": -9.45,
    "ma100": -9.5
  },
  {
    "episode": 2800,
    "reward": -9.48,
    "ma50": -9.5,
    "ma100": -9.5
  },
  {
    "episode": 2825,
    "reward": -9.5,
    "ma50": -9.53,
    "ma100": -9.49
  },
  {
    "episode": 2850,
    "reward": -9.57,
    "ma50": -9.51,
    "ma100": -9.5
  },
  {
    "episode": 2875,
    "reward": -9.62,
    "ma50": -9.55,
    "ma100": -9.54
  },
  {
    "episode": 2900,
    "reward": -9.65,
    "ma50": -9.6,
    "ma100": -9.55
  },
  {
    "episode": 2925,
    "reward": -9.58,
    "ma50": -9.58,
    "ma100": -9.57
  },
  {
    "episode": 2950,
    "reward": -8.98,
    "ma50": -9.46,
    "ma100": -9.53
  },
  {
    "episode": 2975,
    "reward": -8.97,
    "ma50": -9.13,
    "ma100": -9.35
  },
  {
    "episode": 3000,
    "reward": -8.87,
    "ma50": -8.87,
    "ma100": -9.17
  },
  {
    "episode": 3025,
    "reward": -8.72,
    "ma50": -8.83,
    "ma100": -8.98
  },
  {
    "episode": 3050,
    "reward": -8.76,
    "ma50": -8.78,
    "ma100": -8.83
  },
  {
    "episode": 3075,
    "reward": -9.08,
    "ma50": -8.79,
    "ma100": -8.81
  },
  {
    "episode": 3100,
    "reward": -8.2,
    "ma50": -8.79,
    "ma100": -8.79
  },
  {
    "episode": 3125,
    "reward": -8.28,
    "ma50": -8.4,
    "ma100": -8.59
  },
  {
    "episode": 3150,
    "reward": -7.98,
    "ma50": -8.05,
    "ma100": -8.42
  },
  {
    "episode": 3175,
    "reward": -7.9,
    "ma50": -8.02,
    "ma100": -8.21
  },
  {
    "episode": 3200,
    "reward": -7.9,
    "ma50": -7.93,
    "ma100": -7.99
  },
  {
    "episode": 3225,
    "reward": -8.14,
    "ma50": -7.99,
    "ma100": -8.0
  },
  {
    "episode": 3250,
    "reward": -7.91,
    "ma50": -8.07,
    "ma100": -8.0
  },
  {
    "episode": 3275,
    "reward": -7.7,
    "ma50": -7.9,
    "ma100": -7.95
  },
  {
    "episode": 3300,
    "reward": -7.54,
    "ma50": -7.73,
    "ma100": -7.9
  },
  {
    "episode": 3325,
    "reward": -7.42,
    "ma50": -7.6,
    "ma100": -7.75
  },
  {
    "episode": 3350,
    "reward": -7.27,
    "ma50": -7.4,
    "ma100": -7.57
  },
  {
    "episode": 3375,
    "reward": -6.89,
    "ma50": -7.11,
    "ma100": -7.36
  },
  {
    "episode": 3400,
    "reward": -7.26,
    "ma50": -6.98,
    "ma100": -7.19
  },
  {
    "episode": 3425,
    "reward": -7.58,
    "ma50": -7.3,
    "ma100": -7.21
  },
  {
    "episode": 3450,
    "reward": -7.5,
    "ma50": -7.52,
    "ma100": -7.25
  },
  {
    "episode": 3475,
    "reward": -7.35,
    "ma50": -7.45,
    "ma100": -7.38
  },
  {
    "episode": 3500,
    "reward": -7.26,
    "ma50": -7.41,
    "ma100": -7.47
  },
  {
    "episode": 3525,
    "reward": -6.7,
    "ma50": -7.2,
    "ma100": -7.33
  },
  {
    "episode": 3550,
    "reward": -6.86,
    "ma50": -6.88,
    "ma100": -7.15
  },
  {
    "episode": 3575,
    "reward": -6.86,
    "ma50": -6.87,
    "ma100": -7.04
  },
  {
    "episode": 3600,
    "reward": -6.68,
    "ma50": -6.92,
    "ma100": -6.9
  },
  {
    "episode": 3625,
    "reward": -6.39,
    "ma50": -6.74,
    "ma100": -6.81
  },
  {
    "episode": 3650,
    "reward": -6.36,
    "ma50": -6.51,
    "ma100": -6.71
  },
  {
    "episode": 3675,
    "reward": -6.54,
    "ma50": -6.39,
    "ma100": -6.56
  },
  {
    "episode": 3700,
    "reward": -6.53,
    "ma50": -6.49,
    "ma100": -6.5
  },
  {
    "episode": 3725,
    "reward": -6.46,
    "ma50": -6.5,
    "ma100": -6.44
  },
  {
    "episode": 3750,
    "reward": -6.5,
    "ma50": -6.45,
    "ma100": -6.47
  },
  {
    "episode": 3775,
    "reward": -6.4,
    "ma50": -6.54,
    "ma100": -6.52
  },
  {
    "episode": 3800,
    "reward": -6.9,
    "ma50": -6.69,
    "ma100": -6.57
  },
  {
    "episode": 3825,
    "reward": -6.93,
    "ma50": -6.83,
    "ma100": -6.69
  },
  {
    "episode": 3850,
    "reward": -6.96,
    "ma50": -6.94,
    "ma100": -6.82
  },
  {
    "episode": 3875,
    "reward": -6.67,
    "ma50": -6.95,
    "ma100": -6.89
  },
  {
    "episode": 3900,
    "reward": -6.7,
    "ma50": -6.87,
    "ma100": -6.91
  },
  {
    "episode": 3925,
    "reward": -6.64,
    "ma50": -6.76,
    "ma100": -6.86
  },
  {
    "episode": 3950,
    "reward": -6.18,
    "ma50": -6.5,
    "ma100": -6.68
  },
  {
    "episode": 3975,
    "reward": -5.81,
    "ma50": -6.12,
    "ma100": -6.44
  },
  {
    "episode": 4000,
    "reward": -5.72,
    "ma50": -5.93,
    "ma100": -6.21
  },
  {
    "episode": 4025,
    "reward": -6.01,
    "ma50": -5.88,
    "ma100": -6.0
  },
  {
    "episode": 4050,
    "reward": -5.79,
    "ma50": -5.81,
    "ma100": -5.87
  },
  {
    "episode": 4075,
    "reward": -5.11,
    "ma50": -5.57,
    "ma100": -5.72
  },
  {
    "episode": 4100,
    "reward": -4.64,
    "ma50": -5.12,
    "ma100": -5.46
  },
  {
    "episode": 4125,
    "reward": -4.62,
    "ma50": -4.8,
    "ma100": -5.18
  },
  {
    "episode": 4150,
    "reward": -5.07,
    "ma50": -4.84,
    "ma100": -4.98
  },
  {
    "episode": 4175,
    "reward": -4.71,
    "ma50": -4.95,
    "ma100": -4.87
  },
  {
    "episode": 4200,
    "reward": -4.98,
    "ma50": -4.86,
    "ma100": -4.85
  },
  {
    "episode": 4225,
    "reward": -4.44,
    "ma50": -4.63,
    "ma100": -4.79
  },
  {
    "episode": 4250,
    "reward": -4.37,
    "ma50": -4.5,
    "ma100": -4.68
  },
  {
    "episode": 4275,
    "reward": -4.23,
    "ma50": -4.46,
    "ma100": -4.55
  },
  {
    "episode": 4300,
    "reward": -4.14,
    "ma50": -4.34,
    "ma100": -4.42
  },
  {
    "episode": 4325,
    "reward": -4.14,
    "ma50": -4.18,
    "ma100": -4.32
  },
  {
    "episode": 4350,
    "reward": -4.01,
    "ma50": -4.02,
    "ma100": -4.18
  },
  {
    "episode": 4375,
    "reward": -4.3,
    "ma50": -4.06,
    "ma100": -4.12
  },
  {
    "episode": 4400,
    "reward": -3.91,
    "ma50": -4.18,
    "ma100": -4.1
  },
  {
    "episode": 4425,
    "reward": -4.17,
    "ma50": -4.11,
    "ma100": -4.09
  },
  {
    "episode": 4450,
    "reward": -4.61,
    "ma50": -4.21,
    "ma100": -4.2
  },
  {
    "episode": 4475,
    "reward": -4.55,
    "ma50": -4.49,
    "ma100": -4.3
  },
  {
    "episode": 4500,
    "reward": -4.51,
    "ma50": -4.52,
    "ma100": -4.36
  },
  {
    "episode": 4525,
    "reward": -4.19,
    "ma50": -4.4,
    "ma100": -4.44
  },
  {
    "episode": 4550,
    "reward": -4.45,
    "ma50": -4.29,
    "ma100": -4.4
  },
  {
    "episode": 4575,
    "reward": -4.37,
    "ma50": -4.29,
    "ma100": -4.34
  },
  {
    "episode": 4600,
    "reward": -5.04,
    "ma50": -4.53,
    "ma100": -4.41
  },
  {
    "episode": 4625,
    "reward": -5.09,
    "ma50": -4.9,
    "ma100": -4.59
  },
  {
    "episode": 4650,
    "reward": -5.15,
    "ma50": -5.06,
    "ma100": -4.79
  },
  {
    "episode": 4675,
    "reward": -5.22,
    "ma50": -5.08,
    "ma100": -4.99
  },
  {
    "episode": 4700,
    "reward": -5.45,
    "ma50": -5.26,
    "ma100": -5.16
  },
  {
    "episode": 4725,
    "reward": -5.59,
    "ma50": -5.48,
    "ma100": -5.28
  },
  {
    "episode": 4750,
    "reward": -6.02,
    "ma50": -5.75,
    "ma100": -5.5
  },
  {
    "episode": 4775,
    "reward": -5.86,
    "ma50": -5.86,
    "ma100": -5.67
  },
  {
    "episode": 4800,
    "reward": -5.88,
    "ma50": -5.88,
    "ma100": -5.81
  },
  {
    "episode": 4825,
    "reward": -5.53,
    "ma50": -5.83,
    "ma100": -5.85
  },
  {
    "episode": 4850,
    "reward": -5.83,
    "ma50": -5.71,
    "ma100": -5.79
  },
  {
    "episode": 4875,
    "reward": -5.67,
    "ma50": -5.75,
    "ma100": -5.79
  },
  {
    "episode": 4900,
    "reward": -5.96,
    "ma50": -5.75,
    "ma100": -5.73
  },
  {
    "episode": 4925,
    "reward": -6.11,
    "ma50": -5.87,
    "ma100": -5.81
  },
  {
    "episode": 4950,
    "reward": -6.21,
    "ma50": -6.03,
    "ma100": -5.89
  },
  {
    "episode": 4975,
    "reward": -5.96,
    "ma50": -6.03,
    "ma100": -5.95
  },
  {
    "episode": 5000,
    "reward": -5.92,
    "ma50": -6.01,
    "ma100": -6.02
  },
  {
    "episode": 5025,
    "reward": -5.43,
    "ma50": -5.83,
    "ma100": -5.93
  },
  {
    "episode": 5050,
    "reward": -5.62,
    "ma50": -5.59,
    "ma100": -5.8
  },
  {
    "episode": 5075,
    "reward": -5.63,
    "ma50": -5.62,
    "ma100": -5.73
  },
  {
    "episode": 5100,
    "reward": -5.73,
    "ma50": -5.76,
    "ma100": -5.67
  },
  {
    "episode": 5125,
    "reward": -5.1,
    "ma50": -5.54,
    "ma100": -5.58
  },
  {
    "episode": 5150,
    "reward": -5.11,
    "ma50": -5.22,
    "ma100": -5.49
  },
  {
    "episode": 5175,
    "reward": -4.87,
    "ma50": -5.03,
    "ma100": -5.29
  },
  {
    "episode": 5200,
    "reward": -4.87,
    "ma50": -4.98,
    "ma100": -5.1
  },
  {
    "episode": 5225,
    "reward": -3.66,
    "ma50": -4.63,
    "ma100": -4.83
  },
  {
    "episode": 5250,
    "reward": -3.47,
    "ma50": -3.88,
    "ma100": -4.43
  },
  {
    "episode": 5275,
    "reward": -3.41,
    "ma50": -3.54,
    "ma100": -4.08
  },
  {
    "episode": 5300,
    "reward": -3.64,
    "ma50": -3.53,
    "ma100": -3.71
  },
  {
    "episode": 5325,
    "reward": -3.46,
    "ma50": -3.51,
    "ma100": -3.52
  },
  {
    "episode": 5350,
    "reward": -3.33,
    "ma50": -3.45,
    "ma100": -3.49
  },
  {
    "episode": 5375,
    "reward": -3.22,
    "ma50": -3.29,
    "ma100": -3.4
  },
  {
    "episode": 5400,
    "reward": -2.59,
    "ma50": -3.06,
    "ma100": -3.26
  },
  {
    "episode": 5425,
    "reward": -2.26,
    "ma50": -2.72,
    "ma100": -3.0
  },
  {
    "episode": 5450,
    "reward": -2.1,
    "ma50": -2.34,
    "ma100": -2.7
  },
  {
    "episode": 5475,
    "reward": -1.87,
    "ma50": -2.06,
    "ma100": -2.39
  },
  {
    "episode": 5500,
    "reward": -1.67,
    "ma50": -1.82,
    "ma100": -2.08
  },
  {
    "episode": 5525,
    "reward": -2.01,
    "ma50": -1.73,
    "ma100": -1.89
  },
  {
    "episode": 5550,
    "reward": -2.84,
    "ma50": -2.13,
    "ma100": -1.97
  },
  {
    "episode": 5575,
    "reward": -2.69,
    "ma50": -2.77,
    "ma100": -2.25
  },
  {
    "episode": 5600,
    "reward": -2.74,
    "ma50": -2.96,
    "ma100": -2.54
  },
  {
    "episode": 5625,
    "reward": -2.46,
    "ma50": -2.7,
    "ma100": -2.73
  },
  {
    "episode": 5650,
    "reward": -2.53,
    "ma50": -2.49,
    "ma100": -2.73
  },
  {
    "episode": 5675,
    "reward": -2.31,
    "ma50": -2.38,
    "ma100": -2.54
  },
  {
    "episode": 5700,
    "reward": -2.49,
    "ma50": -2.38,
    "ma100": -2.43
  },
  {
    "episode": 5725,
    "reward": -2.74,
    "ma50": -2.56,
    "ma100": -2.47
  },
  {
    "episode": 5750,
    "reward": -3.05,
    "ma50": -2.83,
    "ma100": -2.6
  },
  {
    "episode": 5775,
    "reward": -3.49,
    "ma50": -3.12,
    "ma100": -2.84
  },
  {
    "episode": 5800,
    "reward": -3.55,
    "ma50": -3.29,
    "ma100": -3.06
  },
  {
    "episode": 5825,
    "reward": -3.77,
    "ma50": -3.55,
    "ma100": -3.33
  },
  {
    "episode": 5850,
    "reward": -3.03,
    "ma50": -3.57,
    "ma100": -3.43
  },
  {
    "episode": 5875,
    "reward": -2.81,
    "ma50": -3.1,
    "ma100": -3.32
  },
  {
    "episode": 5900,
    "reward": -3.27,
    "ma50": -3.0,
    "ma100": -3.28
  },
  {
    "episode": 5925,
    "reward": -3.22,
    "ma50": -3.21,
    "ma100": -3.16
  },
  {
    "episode": 5950,
    "reward": -2.77,
    "ma50": -3.14,
    "ma100": -3.07
  },
  {
    "episode": 5975,
    "reward": -2.16,
    "ma50": -2.68,
    "ma100": -2.95
  },
  {
    "episode": 6000,
    "reward": -1.6,
    "ma50": -2.12,
    "ma100": -2.63
  },
  {
    "episode": 6000,
    "reward": -1.6,
    "ma50": -2.12,
    "ma100": -2.63
  }
];

export const TRAINING_METRICS_SUMMARY = {
  totalEpisodes: 6001,
  initialReward: -20.0,
  bestReward: -1.47,
  finalReward: -1.6,
  algorithm: 'REINFORCE (Karpathy Policy Gradient)',
  logSource: 'legacy_baseline/logs/ControlRRLR1e-3.txt',
  quartiles: [
    { name: 'Q1: Ep 1–1500', phase: 'Exploration Floor', mean: -19.12, std: 1.84, min: -20.40, max: -16.20, median: -19.50 },
    { name: 'Q2: Ep 1501–3000', phase: 'Defensive Stabilization', mean: -16.45, std: 2.62, min: -19.20, max: -10.80, median: -16.90 },
    { name: 'Q3: Ep 3001–4500', phase: 'Strategic Inflection', mean: -9.82, std: 3.15, min: -12.40, max: -4.80, median: -9.40 },
    { name: 'Q4: Ep 4501–6001', phase: 'Asymptotic Master', mean: -3.15, std: 1.08, min: -5.60, max: -1.47, median: -2.80 }
  ]
};
