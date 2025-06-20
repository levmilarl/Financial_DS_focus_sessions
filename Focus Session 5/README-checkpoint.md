# VaR and ES Problem Set with Kalman Filter and MLE

## Overview
This repository contains a problem set for B.Sc. students focused on computing Value-at-Risk (VaR) and Expected Shortfall (ES) using Maximum Likelihood Estimation (MLE) and Kalman Filter techniques. The problem set is based on stochastic volatility modeling and includes comparisons with EGARCH models.

## File Structure

### Student Materials
- `KF_ProblemSet_Student.ipynb` - First part of the problem set for students
- `KF_ProblemSet_Student_Part2.ipynb` - Second part of the problem set for students



## Problem Set Structure

The problem set is divided into several parts:

1. **Simulate Data from a Stochastic Volatility Model**
   - Initialize and generate a log-volatility process (AR(1))
   - Generate returns with stochastic volatility

2. **Kalman Filter Implementation**
   - Implement the Kalman Filter algorithm for state estimation
   - Understand the distribution of the error term in the linearized model

3. **Maximum Likelihood Estimation**
   - Implement the objective function for MLE
   - Optimize parameters using numerical optimization
   - Run the Kalman Filter with estimated parameters

4. **EGARCH Model Fitting**
   - Fit an EGARCH(1,1) model to the simulated data
   - Extract volatility estimates for comparison

5. **Computing VaR and ES**
   - Implement functions to compute VaR and ES
   - Calculate these measures for both modeling approaches

6. **Model Evaluation**
   - Implement coverage ratio analysis
   - Compare the performance of the Kalman Filter and EGARCH approaches

7. **Visualization**
   - Create comparative plots of true vs. estimated volatility
   - Visualize returns vs. risk measures

## Instructions

### For Students:
1. Work through the problem set notebooks (`KF_ProblemSet_Student.ipynb` and `KF_ProblemSet_Student_Part2.ipynb`)
2. Complete all the tasks marked with "TODO" comments
3. Analyze and interpret the results


## Prerequisites
- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- arch (for EGARCH modeling)
