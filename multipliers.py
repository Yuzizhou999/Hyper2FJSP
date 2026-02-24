"""Multipliers for normalizing the different objectives."""

from enums import ObjectiveFn

MULTIPLIERS_SD2_10x5 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 4.0 / 3.0,
    ObjectiveFn.TOTAL_TARDINESS: 4.0 / 20.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 4.0 / 20.0,
}

MULTIPLIERS_SD2_20x5 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 8.0 / 4.5,
    ObjectiveFn.TOTAL_TARDINESS: 8.0 / 90.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 8.0 / 15.0,
}

MULTIPLIERS_SD2_15x10 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 5.5 / 5.5,
    ObjectiveFn.TOTAL_TARDINESS: 5.5 / 55,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 5.5 / 35.0,
}

MULTIPLIERS_SD2_20x10 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 8.0 / 7.0,
    ObjectiveFn.TOTAL_TARDINESS: 8.0 / 90.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 8.0 / 50.0,
}

MULTIPLIERS_SD1_10x5 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 3.2 / 3.0,
    ObjectiveFn.TOTAL_TARDINESS: 3.2 / 11.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0 / 4.5,
    ObjectiveFn.COSTS: 3.2 / 2.5,
}

MULTIPLIERS_SD1_10x5_SR = {
    ObjectiveFn.MAKESPAN: 140.0 / 140.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 140.0 / 140.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 140.0 / 90.0,
    ObjectiveFn.TOTAL_TARDINESS: 140.0 / 230.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 140.0 / 490.0,
}

MULTIPLIERS_SD1_20x5_SR = {
    ObjectiveFn.MAKESPAN: 240.0 / 240.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 240.0 / 240.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 240.0 / 140.0,
    ObjectiveFn.TOTAL_TARDINESS: 240.0 / 1902.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 240.0 / 940.0,
}

MULTIPLIERS_SD1_15x10_SR = {
    ObjectiveFn.MAKESPAN: 210.0 / 210.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 210.0 / 210.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 210.0 / 140.0,
    ObjectiveFn.TOTAL_TARDINESS: 210.0 / 210.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 210.0 / 1500.0,
}

MULTIPLIERS_SD1_20x10_SR = {
    ObjectiveFn.MAKESPAN: 260.0 / 260.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 260.0 / 260.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 260.0 / 175.0,
    ObjectiveFn.TOTAL_TARDINESS: 260.0 / 950.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 260.0 / 1950.0,
}

MULTIPLIERS_SD1_20x5 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 8.5 / 4.5,
    ObjectiveFn.TOTAL_TARDINESS: 8.5 / 80.0,
    ObjectiveFn.TOTAL_EARLINESS: 0.25,
    ObjectiveFn.COSTS: 8.5 / 4.8,
}

MULTIPLIERS_SD1_15x10 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 4.0 / 4.5,
    ObjectiveFn.TOTAL_TARDINESS: 4.0 / 11,
    ObjectiveFn.TOTAL_EARLINESS: 1.0 / 8.0,
    ObjectiveFn.COSTS: 4.0 / 10.0,
}

MULTIPLIERS_SD1_20x10 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 7.0 / 6.0,
    ObjectiveFn.TOTAL_TARDINESS: 7.0 / 45.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0 / 7.0,
    ObjectiveFn.COSTS: 7.0 / 13.5,
}

MULTIPLIERS_JSSP_6x6 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 2.0 / 2.5,
    ObjectiveFn.TOTAL_TARDINESS: 2.0 / 2.6,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 1.0,  # 0
}

MULTIPLIERS_JSSP_10x10 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 4.0 / 5.0,
    ObjectiveFn.TOTAL_TARDINESS: 4.0 / 9.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 1.0,
}

MULTIPLIERS_JSSP_15x15 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 6.5 / 8.1,
    ObjectiveFn.TOTAL_TARDINESS: 6.5 / 25.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 1.0,  # 0
}

MULTIPLIERS_JSSP_20x20 = {
    ObjectiveFn.MAKESPAN: 1.0,
    ObjectiveFn.NEGATIVE_MAKESPAN: 1.0,
    ObjectiveFn.AVERAGE_FLOWTIME: 8.7 / 11.2,
    ObjectiveFn.TOTAL_TARDINESS: 8.7 / 39.0,
    ObjectiveFn.TOTAL_EARLINESS: 1.0,
    ObjectiveFn.COSTS: 1.0,
}

MULTIPLIERS_JSSP_30x20 = MULTIPLIERS_JSSP_20x20
MULTIPLIERS_JSSP_50x20 = MULTIPLIERS_JSSP_20x20
MULTIPLIERS_JSSP_100x20 = MULTIPLIERS_JSSP_20x20

MULTIPLIERS_SD1_30x10 = MULTIPLIERS_SD1_20x10
MULTIPLIERS_SD1_40x10 = MULTIPLIERS_SD1_20x10

MULTIPLIERS_SD2_30x10 = MULTIPLIERS_SD2_20x10
MULTIPLIERS_SD2_40x10 = MULTIPLIERS_SD2_20x10

MULTIPLIERS_BenchData = MULTIPLIERS_SD1_15x10


MULTIPLIERS = {
    ("SD2", 10, 5): MULTIPLIERS_SD2_10x5,
    ("SD2", 20, 5): MULTIPLIERS_SD2_20x5,
    ("SD2", 15, 10): MULTIPLIERS_SD2_15x10,
    ("SD2", 20, 10): MULTIPLIERS_SD2_20x10,
    ("SD1", 10, 5): MULTIPLIERS_SD1_10x5,
    ("SD1", 20, 5): MULTIPLIERS_SD1_20x5,
    ("SD1", 15, 10): MULTIPLIERS_SD1_15x10,
    ("SD1", 20, 10): MULTIPLIERS_SD1_20x10,
    ("SD1", 30, 10): MULTIPLIERS_SD1_30x10,
    ("SD1", 40, 10): MULTIPLIERS_SD1_40x10,
    ("SD2", 30, 10): MULTIPLIERS_SD2_30x10,
    ("SD2", 40, 10): MULTIPLIERS_SD2_40x10,
    "BenchData": MULTIPLIERS_BenchData,
    ("JSSP", 6, 6): MULTIPLIERS_JSSP_6x6,
    ("JSSP", 10, 10): MULTIPLIERS_JSSP_10x10,
    ("JSSP", 15, 15): MULTIPLIERS_JSSP_15x15,
    ("JSSP", 20, 20): MULTIPLIERS_JSSP_20x20,
    ("JSSP", 30, 20): MULTIPLIERS_JSSP_30x20,
    ("JSSP", 50, 20): MULTIPLIERS_JSSP_50x20,
    ("JSSP", 100, 20): MULTIPLIERS_JSSP_100x20,
    ("FFSP", 15, 12): MULTIPLIERS_BenchData,
    ("FFSP", 20, 12): MULTIPLIERS_BenchData,
    ("SD1_SR", 10, 5): MULTIPLIERS_SD1_10x5_SR,
    ("SD1_SR", 20, 5): MULTIPLIERS_SD1_20x5_SR,
    ("SD1_SR", 15, 10): MULTIPLIERS_SD1_15x10_SR,
    ("SD1_SR", 20, 10): MULTIPLIERS_SD1_20x10_SR,
}
