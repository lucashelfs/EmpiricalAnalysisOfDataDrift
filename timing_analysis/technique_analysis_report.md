# Technique Performance Analysis Report

## Technique Rankings

### By Average Speed (Fastest to Slowest)
1. **JSDDM**: 1.879s average
2. **HDDDM**: 2.158s average
3. **KSDDM_90**: 7.298s average
4. **KSDDM_95**: 8.098s average

### By Consistency (Most to Least Consistent)
1. **JSDDM**: 2.607s std deviation
2. **HDDDM**: 3.037s std deviation
3. **KSDDM_90**: 11.018s std deviation
4. **KSDDM_95**: 12.643s std deviation

## Batch Size Recommendations

- **HDDDM**: Best batch size = 2500, Worst = 1000
- **KSDDM_95**: Best batch size = 2500, Worst = 1000
- **KSDDM_90**: Best batch size = 2000, Worst = 1000
- **JSDDM**: Best batch size = 2500, Worst = 1000