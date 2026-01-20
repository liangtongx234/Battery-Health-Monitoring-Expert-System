# NUDT Battery Aging Dataset

## Dataset Description

This dataset contains battery aging data collected at the National University of Defense Technology (NUDT). Due to the large file size , the dataset cannot be hosted directly on GitHub.

**Data Availability:** The dataset is being prepared for archival on [Zenodo](https://zenodo.org/). If you need access to the data before it becomes publicly available, please contact us via email or open an issue in this repository.

## Naming Convention

File naming format: `BatteryID-ChargeState-DischargeState-CycleNumber`

## Experimental Conditions

| Parameter | Value |
|-----------|-------|
| Charging protocol | CC-CV at 1/3 C to 4.25 V |
| Discharging protocol | CC at 2 C to 2.5 V |
| Rest period | 10 min between charge/discharge |

## Notes

- The rest period (10 min) allows the battery to reach thermal and electrochemical equilibrium before the next cycle.
- All tests were conducted under controlled laboratory conditions.

## Contact

For data requests or questions, please contact: liangfuyuan@nudt.edu.cn or open an issue in this repository.