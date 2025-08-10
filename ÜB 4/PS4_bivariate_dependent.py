# Import libraries
import os
import time
import pickle
import multiprocessing as mp
import pandas as pd
import numpy as np
from loguru import logger
from numpy.ma.core import product
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Directories directly relevant to this script
baseFolder = 'courseMaterialPortfolioSorts'  # Project folder, relevant data is stored here
logFilePath = baseFolder + '/logfilePortfolioSortsPipeline.log'  # The log file goes here
resultsPath = baseFolder + '/solution'  # Solutions go here

# Parameters for parallelization (warning: check RAM load on your machine)
num_processes = 16  # Number of processes that are run in parallel
chunk_size = 8  # number of objects that each process works on

# Other important parameters
startDate = pd.to_datetime('01.01.2005', format='%d.%m.%Y')  # start date for the procedure
endDate = pd.to_datetime('01.01.2023', format='%d.%m.%Y')  # end date for the procedure
sortingType = "BivariateIndependent"  # alternatively "BivariateDependent" or "BivariateIndependent"
marketCapColumn = "MCap"  # name of the column that contains the market capitalization
sortVariable1 = "bakshi_mu_tau_30"  # name of the first sorting variable
sortVariable2 = "bakshiSkew_tau_30"  # name of the second sorting variable
#sortVariable1 = "P_beta"  # name of the first sorting variable
#sortVariable2 = "MCap"  # name of the second sorting variable

outcomeVariable = "return_month_ahead_excess"  # name of the outcome variable
sortPercentilesVar1 = [25, 50, 75]  # percentiles for the split of sort variable 1
sortPercentilesVar2 = [25, 50, 75]  # percentiles for the split of sort variable 2
weightType = "MarketCapWeighted"  # weighting scheme within the portfolios, alternatively "EquallyWeighted"
filterPortfolios = None  # Optional parameter for filtering out sorted portfolios by index
backtrackingDays = 0  # Variable to handle the number of days of backtracking for portfolio sorts, values other than 0 are not supported in this code version
lags = 6  # the number of lags for the Newey-West adjustment for the statistical tests
testmode = False  # if test mode is set to true, the sorting of portfolios will only be applied to a small number of dates
log_results = False  # Variable to control whether the results of the portfolio sorting are printed in the log


# define class that handles all computations and logic
class PortfolioSortsPipeline():

    def runPipeline(self):
        # This function is where all steps of the procedure are called one after the other
        # Set up logging
        start_time = time.time()

        logger.add(logFilePath, level='INFO', mode='w')

        logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        logger.info('Start portfolio sorts pipeline...')
        # -----------------
        # Each function performs a step in the procedure
        self.calcMonthlyExcessReturn()
        self.mergeData()
        self.getSortingDates()
        self.buildSortedPortfolios()
        self.convertOutcome()
        self.calcDiffPortfolios()
        self.performStatTest()
        # -----------------
        end_time = time.time()
        duration_seconds = end_time - start_time
        duration_formatted = time.strftime('%H:%M:%S', time.gmtime(duration_seconds))
        logger.info('Portfolio sorts pipeline done!')
        logger.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        logger.info(f'Duration: {duration_formatted}')

    def calcMonthlyExcessReturn(self):
        # File Paths
        prices_filepath = baseFolder + '/SPXConstituentsPrices.parquet'
        riskfree_filepath = baseFolder + '/RiskFreeRatesFiltered.parquet'
        riskfree_output = resultsPath + "/RiskFreeRatesFilteredMonthly.parquet"
        prices_output = resultsPath + "/SPXConstituentsPricesPrepared.parquet"

        # Load Prices and Risk-Free Rates
        pricesData = pd.read_parquet(prices_filepath)
        riskFreeRatesData = pd.read_parquet(riskfree_filepath)

        # Convert 'date' columns to pd.Timestamp for both dataframes
        pricesData['date'] = pd.to_datetime(pricesData['date'])
        riskFreeRatesData['date'] = pd.to_datetime(riskFreeRatesData['date'])

        ### A - Risk Free Rates ###
        # Filter for daystomaturity = 30 and drop unnecessary columns
        ########## STUDENT TASK START ##########
        riskFreeRatesData = riskFreeRatesData[riskFreeRatesData['daystomaturity'] == 30]
        ########## STUDENT TASK END ##########
        riskFreeRatesData = riskFreeRatesData.drop(columns=['yld_pct_annual', 'yld_pct_daily'])

        # Save the filtered risk-free rates
        riskFreeRatesData.to_parquet(riskfree_output, index=False)
        logger.info(f"Processed risk free rates saved to {riskfree_output}")

        ### B - Prices Data Processing ###
        # Drop unnecessary columns
        cols_to_drop = ['bidlow', 'askhigh', 'volume', 'totalreturn', 'adjustmentfactor', 'openprice',
                        'sharesoutstanding']
        ########## STUDENT TASK START ##########
        pricesData = pricesData.drop(columns=cols_to_drop)
        ########## STUDENT TASK END ##########

        # Filter by date range
        pricesData = pricesData[(pricesData['date'] >= startDate) & (pricesData['date'] <= endDate)]

        # Adjusted Close Price Calculation
        pricesData['adjustedclose'] = (pricesData['closeprice'] * pricesData['adjustmentfactor2']) / \
                                      pricesData.groupby('securityid')['adjustmentfactor2'].transform('last')

        # Monthly Discrete Returns (20-trading-day rolling return)
        pricesData['return_month'] = pricesData.groupby('securityid')['adjustedclose'].pct_change(periods=20)

        # Merge with Risk-Free Rates on 'date'
        pricesData = pricesData.merge(riskFreeRatesData, on='date', how='left')

        # Calculate Monthly Excess Return
        pricesData['return_month_excess'] = pricesData['return_month'] - pricesData['yld_pct_monthly']

        # Offset Monthly Excess Return by 20 trading days into the past
        ########## STUDENT TASK START ##########
        pricesData['return_month_ahead_excess'] = pricesData.groupby('securityid')['return_month_excess'].shift(-20)
        ########## STUDENT TASK END ##########

        # Drop some columns
        pricesData = pricesData.drop(columns=['closeprice', 'adjustmentfactor2', 'daystomaturity', 'yld_pct_monthly'])

        # Save the final processed dataset
        pricesData.to_parquet(prices_output, index=False)
        logger.info(f"Processed data saved to {prices_output}")

    def mergeData(self):
        # File paths for the data
        prices_filepath = resultsPath + "/SPXConstituentsPricesPrepared.parquet"
        cram_filepath = baseFolder + "/CRAMMeasures.parquet"
        merged_output = resultsPath + "/MergedInputData.parquet"

        # Load the data
        pricesData = pd.read_parquet(prices_filepath)
        cramData = pd.read_parquet(cram_filepath)

        # Ensure 'date' in pricesData and 'loctimestamp' in cramData are in the same format
        ########## STUDENT TASK START ##########
        pricesData['date'] = pd.to_datetime(pricesData['date'])
        cramData['loctimestamp'] = pd.to_datetime(cramData['loctimestamp'])
        ########## STUDENT TASK END ##########

        # Filter CRAM data by date
        cramData = cramData[(cramData['loctimestamp'] >= startDate) & (cramData['loctimestamp'] <= endDate)]

        # Merge the data on SecurityID and date/loctimestamp
        mergedData = pd.merge(
            cramData,
            pricesData[['securityid', 'date', 'return_month_ahead_excess']],
            how='left',
            left_on=['SecurityID', 'loctimestamp'],
            right_on=['securityid', 'date']
        )

        # Drop the extra 'date' column that was added from pricesData
        ########## STUDENT TASK START ##########
        mergedData = mergedData.drop(columns=['date'])
        ########## STUDENT TASK END ##########

        # Save the merged data to a new Parquet file
        mergedData.to_parquet(merged_output, index=False)

        logger.info(f"Merged data saved to {merged_output}")

    def getSortingDates(self):

        # File paths
        merged_filepath = resultsPath + "/MergedInputData.parquet"
        filtered_output = resultsPath + "/MergedInputDataFiltered.parquet"
        sorting_dates_output = resultsPath + "/SortingDates.parquet"

        # Load merged data
        mergedData = pd.read_parquet(merged_filepath)

        # Ensure 'loctimestamp' is a datetime object
        mergedData['loctimestamp'] = pd.to_datetime(mergedData['loctimestamp'])

        # Extract unique dates
        unique_dates = sorted(mergedData['loctimestamp'].unique())  # Sort in ascending order

        # Create a dataframe from the unique dates
        unique_dates_df = pd.DataFrame({'loctimestamp': unique_dates})

        # Find the first available date for each month
        unique_dates_df['year_month'] = unique_dates_df['loctimestamp'].dt.to_period('M')  # Extract year-month
        first_of_month_dates = unique_dates_df.groupby('year_month')['loctimestamp'].min().reset_index(drop=True)

        # Filter original dataframe to keep only rows with these first-of-month dates
        ########## STUDENT TASK START ##########
        filteredData = mergedData[mergedData['loctimestamp'].isin(first_of_month_dates)]
        ########## STUDENT TASK END ##########

        # Reset index before saving
        filteredData = filteredData.reset_index(drop=True)
        first_of_month_dates = first_of_month_dates.reset_index(drop=True)

        # Save the filtered dataframe
        filteredData.to_parquet(filtered_output, index=False)

        # Save the list of first-of-month dates
        first_of_month_dates.to_frame(name="first_of_month").to_parquet(sorting_dates_output, index=False)

        logger.info(f"Filtered dataset saved to {filtered_output}")
        logger.info(f"Sorting dates saved to {sorting_dates_output}")

    def buildSortedPortfoliosChild(self, instrumentData, dateChunk, entitiesByDates, sortingType, marketCapColumn,
                                   sortVariable1, sortVariable2, outcomeVariable, sortPercentilesVar1,
                                   sortPercentilesVar2, weightType, filterPortfolios, backtrackingDays, resultsQueue,
                                   missingDataQueue):
        # Process a subset of dates and return data of sorted portfolios and encountered errors to the result queues
        pid = os.getpid()  # Variable to store the process ID, to be able to identify the child process
        # logger.info(f'Process {pid} started with {len(dateChunk)} dates.')
        portfolioWeightsDictChild = {}  # Key: portfolioID, Value: {date: weights}
        outcomeResultsDictChild = {}  # Key: portfolioID, Value: {date: averageOutcome}
        breakpointsDictChild = {}  # breakpoints per date
        missingDataDictChild = {}  # Key: date, Value: {errortype: count}
        missingRebalancingDatesListChild = []  # Dates for which no portfolio weights are calculated due to missing data
        allInstruments = instrumentData['instrumentid'].unique().tolist()

        # Function to handle possible duplicate column names when counting errors
        def countNaNs(df, colname):
            col = df[colname]
            if isinstance(col, pd.DataFrame):  # If column name, i.e. variable, comes up more than once
                return col.iloc[:, 0].isna().sum()
            return col.isna().sum()

        # Iterate over each date and perform sorting
        for date in dateChunk:
            workingDate = date
            validDataFound = False
            backtrackAttempts = 0
            missingData = {
                date: {"instruments_missing": 0, "sortVariable1_nan": 0, "sortVariable2_nan": 0, "marketCap_nan": 0,
                       "outcomeVar_nan": 0, "portfolio_insufficient": 0}}
            if entitiesByDates is not None and not entitiesByDates.empty:
                constituents = entitiesByDates[entitiesByDates['Date'] == date]['InstrumentID']
                ####################################################################################################################################################################################
                # Duplicate instrumentIDs should never occur!
                # Fix this somewhere else, i.e in the mapping, then remove this code block!
                # Look at this InsID: USOPT0003356D1
                duplicateIDs = constituents[
                    constituents.duplicated(keep=False)]  # Check for duplicates in InstrumentIDs
                if not duplicateIDs.empty:
                    logger.error(
                        f'Process {pid}: Dropping duplicate InstrumentIDs found on {date}: {duplicateIDs.unique()}')
                constituents = constituents.drop_duplicates()
                ####################################################################################################################################################################################
            else:
                constituents = allInstruments
            if sortingType == "Univariate":
                while not validDataFound and backtrackAttempts <= backtrackingDays:
                    # Try to find matching data
                    dateDataframe = instrumentData[  # filter the dataframe in a vectorized manner
                        (instrumentData['loctimestamp'] == workingDate) &
                        (instrumentData['instrumentid'].isin(constituents))
                        ][['instrumentid', sortVariable1, marketCapColumn, outcomeVariable]]
                    validDataframe = dateDataframe.dropna(subset=[sortVariable1, marketCapColumn, outcomeVariable])
                    instrumentDataValid = list(validDataframe.itertuples(index=False, name=None))  # Get instrument data
                    # Step 1: Determine values of the sort variable
                    ########## STUDENT TASK START ##########
                    sort1Values = [x[1] for x in instrumentDataValid]
                    ########## STUDENT TASK END ##########
                    if len(sort1Values) > 0:
                        validDataFound = True
                    else:
                        logger.warning(
                            f"{pid} No valid CRAM data found for {workingDate}. Backtracking to the previous date.")
                        backtrackAttempts += 1
                        workingDate = workingDate - pd.Timedelta(days=1)
                if backtrackAttempts > backtrackingDays:
                    logger.error(
                        f"{pid} ERROR Failed to find valid CRAM data for {date} after {backtrackingDays} attempts.")
                    missingRebalancingDatesListChild.append(date)
                    continue  # Stop searching and log error for the original date.
                # Count instruments with no data
                missingIDs = set(constituents) - set(dateDataframe['instrumentid'])
                missingData[date]["instruments_missing"] += len(missingIDs)
                # Count NaNs
                missingData[date]["sortVariable1_nan"] += int(countNaNs(dateDataframe, sortVariable1))
                missingData[date]["marketCap_nan"] += int(countNaNs(dateDataframe, marketCapColumn))
                missingData[date]["outcomeVar_nan"] += int(countNaNs(dateDataframe, outcomeVariable))
                missingDataDictChild[date] = missingData[date]
                ########## STUDENT TASK START ##########
                breakpoints = np.percentile(sort1Values, sortPercentilesVar1)  # Step 2: Calculate breakpoints
                breakpoints = [float("-inf")] + list(breakpoints) + [float("inf")]
                ########## STUDENT TASK END ##########
                portfolios = {i + 1: [] for i in
                              range(len(breakpoints) - 1)}  # Initialize dictionary structure to put portfolios in
                for instrumentID, sort1Value, marketCap, outcomeValue in instrumentDataValid:  # Step 3: Sort instruments based on breakpoints
                    ########## STUDENT TASK START ##########
                    for i in range(len(breakpoints) - 1):
                        if breakpoints[i] <= sort1Value <= breakpoints[i + 1]:
                            portfolios[i + 1].append((instrumentID, sort1Value, marketCap, outcomeValue))
                            # Notice: No break of the loop, the instrument can be in multiple portfolios
                    ########## STUDENT TASK END ##########

                for portfolioID, instruments in portfolios.items():  # remove duplicate instruments for each portfolio
                    portfolios[portfolioID] = list({x[0]: x for x in instruments}.values())

                # Check for duplicates in each portfolio
                for portfolioID, instruments in portfolios.items():
                    instrumentIDs = [x[0] for x in instruments]  # Extract instrument IDs
                    if len(set(instrumentIDs)) != len(instrumentIDs):
                        duplicateIDs = [x for x in instrumentIDs if instrumentIDs.count(x) > 1]
                        logger.error(f"Duplicates found in portfolio {portfolioID} on date {date}: {set(duplicateIDs)}")
                        assert False, f"Duplicates in portfolio {portfolioID} on date {date}"

                # Step 4: Filter portfolios (optional)
                if filterPortfolios:
                    filteredPortfolios = {f"{sortVariable1}_" + str(i): portfolios[i] for i in filterPortfolios if
                                          i in portfolios}
                else:
                    filteredPortfolios = {f"{sortVariable1}_" + str(i): portfolios[i] for i in portfolios}

                # Step 5: Calculate Weights & average outcome variable
                portfolioWeights = {}
                portfolioOutcome = {}
                for portfolioID, instruments in filteredPortfolios.items():
                    ########## STUDENT TASK END ##########
                    if weightType == "EquallyWeighted":
                        weight = 1 / len(instruments)
                        portfolioWeights[portfolioID] = {instrumentID: weight for instrumentID, _, _, _ in
                                                         instruments}  # weights for each instrument in the portfolio
                        weightedSum = sum(weight * outcomeValue for _, _, _, outcomeValue in
                                          instruments)  # weighted outcome variable for each portfolio
                    elif weightType == "MarketCapWeighted":
                        totalMarketCap = sum(marketCap for _, _, marketCap, _ in instruments)
                        portfolioWeights[portfolioID] = {instrumentID: marketCap / totalMarketCap for
                                                         instrumentID, _, marketCap, _ in instruments}
                        weightedSum = sum(
                            (marketCap / totalMarketCap) * outcomeValue for _, _, marketCap, outcomeValue in
                            instruments)
                    portfolioOutcome[portfolioID] = weightedSum
                    ########## STUDENT TASK END ##########

                # Collect results
                for portfolioID, weightStructure in portfolioWeights.items():
                    portfolioWeightsAtDate = {date: weightStructure}
                    if portfolioID not in portfolioWeightsDictChild:
                        portfolioWeightsDictChild[portfolioID] = {}  # Initialize portfolioID entry
                    for date, weights in portfolioWeightsAtDate.items():
                        portfolioWeightsDictChild[portfolioID][date] = weights  # Update with new data

                for portfolioID, outcome in portfolioOutcome.items():
                    portfolioOutcomeAtDate = {date: outcome}
                    if portfolioID not in outcomeResultsDictChild:
                        outcomeResultsDictChild[portfolioID] = {}  # Initialize portfolioID entry
                    for date, outcome in portfolioOutcomeAtDate.items():
                        outcomeResultsDictChild[portfolioID][date] = outcome

                # Set up breakpoint dictionary
                if date not in breakpointsDictChild.keys():
                    breakpointsDictChild[date] = {}
                breakpointsDictChild[date][sortVariable1] = breakpoints

            elif sortingType == "BivariateDependent":
                ########## STUDENT TASK LEVEL 2 START ##########

                while not validDataFound and backtrackAttempts <= backtrackingDays:
                    # Try to find matching data
                    dateDataframe = instrumentData[  # filter the dataframe in a vectorized manner
                        (instrumentData['loctimestamp'] == workingDate) &
                        (instrumentData['instrumentid'].isin(constituents))
                        ][['instrumentid', sortVariable1, sortVariable2, marketCapColumn, outcomeVariable]]

                    validDataframe = dateDataframe.dropna(subset=[sortVariable1, sortVariable2, marketCapColumn, outcomeVariable])
                    instrumentDataValid = list(validDataframe.itertuples(index=False, name=None))  # Get instrument data

                    # Step 1: Determine values of the first sort variable
                    sort1Values = [x[1] for x in instrumentDataValid]
                    if len(sort1Values) > 0:
                        validDataFound = True
                    else:
                        logger.warning(
                            f"{pid} No valid CRAM data found for {workingDate}. Backtracking to the previous date.")
                        backtrackAttempts += 1
                        workingDate = workingDate - pd.Timedelta(days=1)

                if backtrackAttempts > backtrackingDays:
                    logger.error(
                        f"{pid} ERROR Failed to find valid CRAM data for {date} after {backtrackingDays} attempts.")
                    missingRebalancingDatesListChild.append(date)
                    continue  # Stop searching and log error for the original date.

                # Count instruments with no data
                missingIDs = set(constituents) - set(dateDataframe['instrumentid'])
                missingData[date]["instruments_missing"] += len(missingIDs)
                # Count NaNs
                missingData[date]["sortVariable1_nan"] += int(countNaNs(dateDataframe, sortVariable1))
                missingData[date]["sortVariable2_nan"] += int(countNaNs(dateDataframe, sortVariable2))
                missingData[date]["marketCap_nan"] += int(countNaNs(dateDataframe, marketCapColumn))
                missingData[date]["outcomeVar_nan"] += int(countNaNs(dateDataframe, outcomeVariable))
                missingDataDictChild[date] = missingData[date]

                x1_breakpoints = np.percentile(sort1Values, sortPercentilesVar1)  # Step 2: Calculate x1 breakpoints
                x1_breakpoints = [float("-inf")] + list(x1_breakpoints) + [float("inf")]
                x2_breakpoints = []  # Initialize x2 breakpoints

                bivariate_portfolios = {} # keys (i,j) each representing a portfolio, values are lists of instruments

                for i in range(len(x1_breakpoints) - 1):
                    instruments_in_x1_bin = [x for x in instrumentDataValid if x1_breakpoints[i] <= x[1] <= x1_breakpoints[i + 1]]
                    if not instruments_in_x1_bin:
                        continue # skip empty bins

                    # get x2 values in each bin
                    x2_values = [x[2] for x in instruments_in_x1_bin]

                    # compute breakpoints for x2 within the current x1 bin
                    x2_breakpoints = np.percentile(x2_values, sortPercentilesVar2)
                    x2_breakpoints = [float("-inf")] + list(x2_breakpoints) + [float("inf")]

                    # assign instruments to bivariate portfolio matrix
                    for instrument in instruments_in_x1_bin:
                        instrumentID, sort1Value, sort2Value, marketCap, outcomeValue = instrument
                        for j in range(len(x2_breakpoints) - 1):
                            if x2_breakpoints[j] <= sort2Value <= x2_breakpoints[j + 1]:
                                port_id = f"{sortVariable1}_{i + 1}_{sortVariable2}_{j + 1}"
                                if port_id not in bivariate_portfolios:
                                    bivariate_portfolios[port_id] = []
                                bivariate_portfolios[port_id].append((instrumentID, sort1Value, sort2Value, marketCap, outcomeValue)) #for consistency with univariate code, we create a new tuple here
                                # no break here!


                for port_id, instruments in bivariate_portfolios.items():  # remove duplicate instruments for each portfolio
                    bivariate_portfolios[port_id] = list({x[0]: x for x in instruments}.values())

                # Check for duplicates in each portfolio
                for portfolioID, instruments in bivariate_portfolios.items():
                    instrumentIDs = [x[0] for x in instruments]  # Extract instrument IDs
                    if len(set(instrumentIDs)) != len(instrumentIDs):
                        duplicateIDs = [x for x in instrumentIDs if instrumentIDs.count(x) > 1]
                        logger.error(f"Duplicates found in portfolio {portfolioID} on date {date}: {set(duplicateIDs)}")
                        assert False, f"Duplicates in portfolio {portfolioID} on date {date}"

                # Step 4: Filter portfolios (optional)
                if filterPortfolios:
                    filteredPortfolios = {
                        port_id: bivariate_portfolios[port_id]
                        for port_id in filterPortfolios
                        if port_id in bivariate_portfolios
                    }
                else:
                    filteredPortfolios = dict(bivariate_portfolios)

                # Step 5: Calculate Weights & average outcome variable
                portfolioWeights = {}
                portfolioOutcome = {}
                for portfolioID, instruments in bivariate_portfolios.items():

                    if weightType == "EquallyWeighted":
                        weight = 1 / len(instruments)
                        portfolioWeights[portfolioID] = {instrumentID: weight for instrumentID, _, _, _, _ in
                                                         instruments}  # weights for each instrument in the portfolio
                        weightedSum = sum(weight * outcomeValue for _, _, _, _, outcomeValue in
                                          instruments)  # weighted outcome variable for each portfolio
                    elif weightType == "MarketCapWeighted":
                        totalMarketCap = sum(marketCap for _, _, _, marketCap, _ in instruments)
                        portfolioWeights[portfolioID] = {instrumentID: marketCap / totalMarketCap for
                                                         instrumentID, _, _, marketCap, _ in instruments}
                        weightedSum = sum(
                            (marketCap / totalMarketCap) * outcomeValue for _, _, _, marketCap, outcomeValue in
                            instruments)
                    portfolioOutcome[portfolioID] = weightedSum

                # Collect results: weights and outcomes
                for portfolioID, weightStructure in portfolioWeights.items():
                    if portfolioID not in portfolioWeightsDictChild:
                        portfolioWeightsDictChild[portfolioID] = {}
                    portfolioWeightsDictChild[portfolioID][date] = weightStructure  # Use current 'date'

                for portfolioID, outcome in portfolioOutcome.items():
                    if portfolioID not in outcomeResultsDictChild:
                        outcomeResultsDictChild[portfolioID] = {}
                    outcomeResultsDictChild[portfolioID][date] = outcome  # Use current 'date'

                # Save breakpoints: both for X1 and X2
                if date not in breakpointsDictChild:
                    breakpointsDictChild[date] = {}

                breakpointsDictChild[date][sortVariable1] = x1_breakpoints
                breakpointsDictChild[date][sortVariable2] = x2_breakpoints

            elif sortingType == "BivariateIndependent":
                # Bivariate Independent sorting: first sort by sortVariable1, then by sortVariable2
                while not validDataFound and backtrackAttempts <= backtrackingDays:
                    # Try to find matching data
                    dateDataframe = instrumentData[  # filter the dataframe in a vectorized manner
                        (instrumentData['loctimestamp'] == workingDate) &
                        (instrumentData['instrumentid'].isin(constituents))
                        ][['instrumentid', sortVariable1, sortVariable2, marketCapColumn, outcomeVariable]]

                    validDataframe = dateDataframe.dropna(subset=[sortVariable1, sortVariable2, marketCapColumn, outcomeVariable])
                    instrumentDataValid = list(validDataframe.itertuples(index=False, name=None))  # Get instrument data

                    # Step 1: Determine values of sort variables
                    sort1Values = [x[1] for x in instrumentDataValid]
                    sort2Values = [x[2] for x in instrumentDataValid]
                    if len(sort1Values) and len(sort2Values) > 0:
                        validDataFound = True
                    else:
                        logger.warning(
                            f"{pid} No valid CRAM data found for {workingDate}. Backtracking to the previous date.")
                        backtrackAttempts += 1
                        workingDate = workingDate - pd.Timedelta(days=1)

                if backtrackAttempts > backtrackingDays:
                    logger.error(
                        f"{pid} ERROR Failed to find valid CRAM data for {date} after {backtrackingDays} attempts.")
                    missingRebalancingDatesListChild.append(date)
                    continue  # Stop searching and log error for the original date.

                # Count instruments with no data
                missingIDs = set(constituents) - set(dateDataframe['instrumentid'])
                missingData[date]["instruments_missing"] += len(missingIDs)
                # Count NaNs
                missingData[date]["sortVariable1_nan"] += int(countNaNs(dateDataframe, sortVariable1))
                missingData[date]["sortVariable2_nan"] += int(countNaNs(dateDataframe, sortVariable2))
                missingData[date]["marketCap_nan"] += int(countNaNs(dateDataframe, marketCapColumn))
                missingData[date]["outcomeVar_nan"] += int(countNaNs(dateDataframe, outcomeVariable))
                missingDataDictChild[date] = missingData[date]

                x1_breakpoints = np.percentile(sort1Values, sortPercentilesVar1)  # Step 2: Calculate x1 breakpoints
                x1_breakpoints = [float("-inf")] + list(x1_breakpoints) + [float("inf")]

                x2_breakpoints = np.percentile(sort2Values, sortPercentilesVar2)  # Step 3: Calculate x2 breakpoints
                x2_breakpoints = [float("-inf")] + list(x2_breakpoints) + [float("inf")]

                bivariate_portfolios = {}  # Step 4: Initialize bivariate portfolios

                for instrument_id, sort_variable1, sort_variable2, market_cap, outcome_value in instrumentDataValid:
                    # Step 5: Assign instruments to portfolios based on breakpoints
                    for i in range(len(x1_breakpoints) - 1):
                        if x1_breakpoints[i] <= sort_variable1 <= x1_breakpoints[i + 1]: # assign to correct x1 bin
                            for j in range(len(x2_breakpoints) - 1):
                                if x2_breakpoints[j] <= sort_variable2 <= x2_breakpoints[j + 1]: # assign to correct x2 bin
                                    port_id = f"{sortVariable1}_{i + 1}_{sortVariable2}_{j + 1}"
                                    if port_id not in bivariate_portfolios:
                                        bivariate_portfolios[port_id] = []
                                    bivariate_portfolios[port_id].append((instrument_id, sort_variable1, sort_variable2,
                                                                          market_cap,
                                                                          outcome_value))  # for consistency with univariate code, we create a new tuple here
                                    # Notice: No break of the loop, the instrument can be in multiple portfolios

                # - same as in dependent version -

                for port_id, instruments in bivariate_portfolios.items():  # remove duplicate instruments for each portfolio
                    bivariate_portfolios[port_id] = list({x[0]: x for x in instruments}.values())

                # Check for duplicates in each portfolio
                for portfolioID, instruments in bivariate_portfolios.items():
                    instrumentIDs = [x[0] for x in instruments]  # Extract instrument IDs
                    if len(set(instrumentIDs)) != len(instrumentIDs):
                        duplicateIDs = [x for x in instrumentIDs if instrumentIDs.count(x) > 1]
                        logger.error(f"Duplicates found in portfolio {portfolioID} on date {date}: {set(duplicateIDs)}")
                        assert False, f"Duplicates in portfolio {portfolioID} on date {date}"

                # Step 6: Filter portfolios (optional)
                if filterPortfolios:
                    filteredPortfolios = {
                        port_id: bivariate_portfolios[port_id]
                        for port_id in filterPortfolios
                        if port_id in bivariate_portfolios
                    }
                else:
                    filteredPortfolios = dict(bivariate_portfolios)

                # Step 7: Calculate Weights & average outcome variable
                portfolioWeights = {}
                portfolioOutcome = {}
                for portfolioID, instruments in bivariate_portfolios.items():

                    if weightType == "EquallyWeighted":
                        weight = 1 / len(instruments)
                        portfolioWeights[portfolioID] = {instrumentID: weight for instrumentID, _, _, _, _ in
                                                         instruments}  # weights for each instrument in the portfolio
                        weightedSum = sum(weight * outcomeValue for _, _, _, _, outcomeValue in
                                          instruments)  # weighted outcome variable for each portfolio
                    elif weightType == "MarketCapWeighted":
                        totalMarketCap = sum(marketCap for _, _, _, marketCap, _ in instruments)
                        portfolioWeights[portfolioID] = {instrumentID: marketCap / totalMarketCap for
                                                         instrumentID, _, _, marketCap, _ in instruments}
                        weightedSum = sum(
                            (marketCap / totalMarketCap) * outcomeValue for _, _, _, marketCap, outcomeValue in
                            instruments)
                    portfolioOutcome[portfolioID] = weightedSum

                # Collect results: weights and outcomes
                for portfolioID, weightStructure in portfolioWeights.items():
                    if portfolioID not in portfolioWeightsDictChild:
                        portfolioWeightsDictChild[portfolioID] = {}
                    portfolioWeightsDictChild[portfolioID][date] = weightStructure  # Use current 'date'

                for portfolioID, outcome in portfolioOutcome.items():
                    if portfolioID not in outcomeResultsDictChild:
                        outcomeResultsDictChild[portfolioID] = {}
                    outcomeResultsDictChild[portfolioID][date] = outcome  # Use current 'date'

                # Save breakpoints: both for X1 and X2
                if date not in breakpointsDictChild:
                    breakpointsDictChild[date] = {}

                breakpointsDictChild[date][sortVariable1] = x1_breakpoints
                breakpointsDictChild[date][sortVariable2] = x2_breakpoints




        # Pass results to queue
        resultsQueue.put((portfolioWeightsDictChild, breakpointsDictChild, outcomeResultsDictChild))
        missingDataQueue.put((missingDataDictChild, missingRebalancingDatesListChild))
        # logger.info(f'Process {pid} done.')

    def buildSortedPortfolios(self):
        # File paths
        instrument_data = resultsPath + "/MergedInputDataFiltered.parquet"
        instrumentData = pd.read_parquet(instrument_data)
        sorting_dates = resultsPath + "/SortingDates.parquet"
        sortingDates = pd.read_parquet(sorting_dates)
        sortingDates = sortingDates["first_of_month"].tolist()  # convert to list
        if testmode:
            sortingDates = sortingDates[0:5]  # Short list for testing
        entitiesByDates = None  # here one could provide a dataframe to determine date specific constituents for algorithm, not relevant in this code version
        logger.info(f'Starting parallelized calculation of sorted portfolios: {sortingType}')
        logger.info(
            f'Sorting variable 1: {sortVariable1}, sorting variable 2: {sortVariable2}, outcome variable: {outcomeVariable}, percentiles variable 1: {sortPercentilesVar1}, percentiles variable 2: {sortPercentilesVar2}')
        logger.info(
            f'Weighting: {weightType}, filter for portfolios: {filterPortfolios}, maximum possible backtracking: {backtrackingDays} days')
        portfolioWeightsDict = {}  # Key: portfolioID, Value: {date: weights}
        outcomeResultsDict = {}  # Key: portfolioID, Value: {date: averageOutcome}
        numberOfStocksDict = {}  # Number of stocks per portfolio
        breakpointsDict = {}  # Breakpoints per date
        missingDataDict = {}  # Key: date, Value: {errortype: count}
        missingRebalancingDatesList = []  # List containing missing rebalancing dates
        chunkSize = len(
            sortingDates) // num_processes + 1  # Divide dates into chunks and handle each chunk in a separate process
        dateChunks = [sortingDates[i:i + chunkSize] for i in
                      range(0, len(sortingDates), chunkSize)]  # List containing lists (chunks) of dates
        resultsQueue = mp.Queue()  # Process queue which will collect results of portfolio sorts
        missingDataQueue = mp.Queue()  # Second process queue which will collect missing data information
        processes = []  # Empty list to contain child processes
        logger.info(f'Chunk size: {chunkSize} meaning dates per chunk')
        logger.info(f'Total number of chunks: {len(dateChunks)} = processes')
        logger.info(f'Spawning child processes...')
        for dateChunk in dateChunks:  # Each chunk gets passed to one child process
            p = mp.Process(
                target=self.buildSortedPortfoliosChild,
                args=(  # The child process gets passed all relevant parameters
                    instrumentData, dateChunk, entitiesByDates,
                    sortingType, marketCapColumn, sortVariable1, sortVariable2, outcomeVariable,
                    sortPercentilesVar1, sortPercentilesVar2,
                    weightType, filterPortfolios, backtrackingDays,
                    resultsQueue, missingDataQueue))
            processes.append(p)
            p.start()
        # Fetch results from queues
        for _ in range(len(dateChunks)):  # Process results for each chunk
            portfolioWeightsDictChild, breakpointsDictChild, outcomeResultsDictChild = resultsQueue.get()
            missingDataDictChild, missingRebalancingDatesListChild = missingDataQueue.get()
            # Collect portfolio weights
            for portfolioID, weightDict in portfolioWeightsDictChild.items():
                if portfolioID not in portfolioWeightsDict:
                    portfolioWeightsDict[portfolioID] = {}  # Initialize portfolioID entry
                for date, weights in weightDict.items():
                    portfolioWeightsDict[portfolioID][date] = weights  # Update with new data
            # Collect average outcome variables
            for portfolioID, outcomeDict in outcomeResultsDictChild.items():
                if portfolioID not in outcomeResultsDict:
                    outcomeResultsDict[portfolioID] = {}  # Initialize
                for date, outcome in outcomeDict.items():
                    outcomeResultsDict[portfolioID][date] = outcome  # Fill data in
            # Collect missing data stats
            for date, dataInfo in missingDataDictChild.items():
                missingDataDict[date] = dataInfo
            # Collect breakpoints
            for date, dataDict in breakpointsDictChild.items():
                if date not in breakpointsDict:
                    breakpointsDict[date] = {}  # Initialize portfolioID entry
                for variable, breakpoints in dataDict.items():
                    if variable not in breakpointsDict[date]:
                        breakpointsDict[date][variable] = {}  # Initialize date entry
                    breakpointsDict[date][variable] = breakpoints  # Update with new data
            if missingRebalancingDatesListChild:
                missingRebalancingDatesList.append(missingRebalancingDatesListChild)

        for p in processes:  # Wait for all processes to finish
            p.join()

        # Sort portfolio results by date ascending for each portfolio
        sortedPortfolioWeightsDict = {
            portfolioID: dict(sorted(weightsByDate.items()))
            for portfolioID, weightsByDate in portfolioWeightsDict.items()
        }
        # Sort outcome results by date ascending for each portfolio
        sortedOutcomeResultsDict = {
            portfolioID: dict(sorted(outcomeByDate.items()))
            for portfolioID, outcomeByDate in outcomeResultsDict.items()
        }
        # Sort missingData by date ascending
        missingDataDict = dict(sorted(missingDataDict.items(), key=lambda item: item[0]))
        # Sort breakpoints by date ascending
        breakpointsDict = dict(sorted(breakpointsDict.items(), key=lambda item: item[0]))
        # Count number of stocks
        for portfolioID, portfolioWeights in sortedPortfolioWeightsDict.items():
            if portfolioID not in numberOfStocksDict:
                numberOfStocksDict[portfolioID] = {}
            for date, weights in portfolioWeights.items():
                numberOfStocksDict[portfolioID][date] = len(weights)
        ##################### PRINT RESULTS TO LOG ########################
        if log_results:
            logger.info("------Portfolios:-------")
            for portfolio, weightStructure in sortedPortfolioWeightsDict.items():
                logger.info(f'Portfolio {portfolio}:')
                for date, weights in weightStructure.items():
                    logger.info(f'Date {date}: Something')
            logger.info("------Outcome:-------")
            for portfolio, outcomeStruct in sortedOutcomeResultsDict.items():
                logger.info(f'Portfolio {portfolio}:')
                for date, outcome in outcomeStruct.items():
                    logger.info(f'Date {date}: {outcome}')
            logger.info("------Missing Data:------")
            for date, dataInfo in missingDataDict.items():
                logger.info(date)
                logger.info(dataInfo)
            logger.info("------Missing rebalancing dates:------")
            logger.info(missingRebalancingDatesList)
            logger.info("------Number of stocks:------")
            for portfolio, nstocksStruct in numberOfStocksDict.items():
                logger.info(f'Portfolio {portfolio}:')
                for date, nstocks in nstocksStruct.items():
                    logger.info(f'Date {date}: {nstocks}')
            logger.info("------Breakpoints:------")
            for date, struct in breakpointsDict.items():
                logger.info(f'date: {date}')
                for var, bps in struct.items():
                    logger.info(f'variable: {var}')
                    logger.info(f'breakpoints: {bps}')
        # Dictionary of variables to save
        variables_to_save = {
            "sortedPortfolioResults": sortedPortfolioWeightsDict,
            "sortedOutcomeResults": sortedOutcomeResultsDict,
            "numberOfStocks": numberOfStocksDict,
            "breakpointsDict": breakpointsDict,
            "missingDataDict": missingDataDict,
            "missingRebalancingDatesList": missingRebalancingDatesList
        }
        # Loop through dictionary and save each variable as a .pkl file
        for var_name, var_value in variables_to_save.items():
            file_path = os.path.join(resultsPath, f"{var_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(var_value, f)
            logger.info(f"Saved {var_name} to {file_path}")
        logger.info('Completed calculation of sorted portfolios.')

    def convertOutcome(self):
        # Path to the input pickle file
        filtered_output = resultsPath + "/sortedOutcomeResults.pkl"

        # Load the sortedOutcomeResults from pickle
        with open(filtered_output, 'rb') as f:
            sortedOutcomeResults = pickle.load(f)

        # Create a list to store all the dates (we assume the dates are the same for all portfolios)
        ########## STUDENT TASK START ##########
        dates = list(next(iter(sortedOutcomeResults.values())).keys())
        ########## STUDENT TASK END ##########

        # Create a dictionary where keys are portfolio names and values are the corresponding outcome values for each date
        portfolio_data = {}

        for portfolio, outcomeStruct in sortedOutcomeResults.items():
            # Create a list of outcomes for the given portfolio, aligned with the dates
            portfolio_data[portfolio] = [outcomeStruct.get(date, None) for date in dates]

        # Convert the dictionary into a DataFrame
        outcome_df = pd.DataFrame(portfolio_data, index=dates)

        # Reset index so 'date' becomes a column in the DataFrame
        outcome_df.reset_index(inplace=True)
        outcome_df.rename(columns={'index': 'date'}, inplace=True)

        port_columns = [col for col in outcome_df.columns if col != 'date']
        sorted_columns = sorted(port_columns)
        outcome_df = outcome_df[['date'] + sorted_columns]  # Sort columns except 'date'

        # Save the DataFrame to a Parquet file
        outcome_df.to_parquet(resultsPath + "/PortfoliosOutcome.parquet", index=False)
        logger.info("PortfoliosOutcome.parquet has been saved.")

    def calcDiffPortfolios(self):
        # Load the PortfoliosOutcome.parquet file
        portfolios_outcome_filepath = resultsPath + "/PortfoliosOutcome.parquet"
        outcome_df = pd.read_parquet(portfolios_outcome_filepath)

        if sortingType == "Univariate":

            # Determine the portfolio names for the highest and lowest portfolios based on sortVariable1
            high_portfolio = f"{sortVariable1}_{len(sortPercentilesVar1) + 1}"  # Highest portfolio
            low_portfolio = f"{sortVariable1}_1"  # Lowest portfolio

            # Check if the high and low portfolios exist in the columns
            if high_portfolio not in outcome_df.columns or low_portfolio not in outcome_df.columns:
                logger.error(f"Portfolio columns {high_portfolio} or {low_portfolio} not found in the DataFrame.")
                return

            # Add the new column: sortVariable1_Diff (difference between high and low portfolio)
            outcome_df[f"{sortVariable1}_Diff"] = outcome_df[high_portfolio] - outcome_df[low_portfolio]

            # Save the resulting DataFrame to a new Parquet file
            output_filepath = resultsPath + "/PortfoliosOutcomePlusDiff.parquet"
            outcome_df.to_parquet(output_filepath, index=False)

            logger.info(f"Resulting DataFrame with the new 'Diff' column saved to {output_filepath}")

        elif sortingType in ["BivariateDependent", "BivariateIndependent"]:
            ########## STUDENT TASK LEVEL 2 START ##########
            # Determine bin count
            n1 = len(sortPercentilesVar1) + 1
            n2 = len(sortPercentilesVar2) + 1

            # Define the portfolio names for A, B, C, D based on slides
            A = f"{sortVariable1}_1_{sortVariable2}_1"
            B = f"{sortVariable1}_{n1}_{sortVariable2}_1"
            C = f"{sortVariable1}_1_{sortVariable2}_{n2}"
            D = f"{sortVariable1}_{n1}_{sortVariable2}_{n2}"

            # Check if all required columns exist
            missing = [col for col in [A, B, C, D] if col not in outcome_df.columns]
            if missing:
                logger.error(f"Missing columns for DiD calculation: {missing}")
                return

            # Calculate the Difference-in-Differences
            outcome_df["Difference in differences"] = (
                    outcome_df[D] - outcome_df[C] - outcome_df[B] + outcome_df[A]
            )

            # Save the resulting DataFrame
            output_filepath = resultsPath + "/PortfoliosOutcomePlusDiff.parquet"
            outcome_df.to_parquet(output_filepath, index=False)

            logger.info(f"Resulting DataFrame with DiD column saved to {output_filepath}")

    def performStatTest(self):
        # Load the PortfoliosOutcomePlusDiff.parquet file
        portfolios_outcome_filepath = resultsPath + "/PortfoliosOutcomePlusDiff.parquet"
        outcome_df = pd.read_parquet(portfolios_outcome_filepath)

        # Exclude the 'date' column and work with portfolio columns
        portfolio_columns = [col for col in outcome_df.columns if col != 'date']

        # Initialize a list to hold the statistics (Avg, StdErr, t-stat, p-value)
        stats_dict = {
            "Average": [],
            "StdErr": [],
            "t-stat": [],
            "p-value": []
        }

        # Calculate the statistics for each portfolio
        for portfolio in portfolio_columns:
            # Get the time series for the portfolio
            timeseries = outcome_df[portfolio]

            # Calculate the average (mean) of the timeseries
            ########## STUDENT TASK START ##########
            avg = np.mean(timeseries)
            ########## STUDENT TASK END ##########
            stats_dict["Average"].append(avg)

            # Calculate the standard error of the timeseries
            ########## STUDENT TASK START ##########
            std_err = np.std(timeseries, ddof=1) / np.sqrt(len(timeseries))
            ########## STUDENT TASK END ##########
            stats_dict["StdErr"].append(std_err)

            # Perform OLS regression with constant (Newey-West adjustment with 6 lags)
            ########## STUDENT TASK START ##########
            model = OLS(timeseries, add_constant(np.ones(len(timeseries))))  # OLS regression with constant
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
            ########## STUDENT TASK END ##########

            # Extract the t-statistic for the mean
            t_stat = results.tvalues.iloc[0]  # t-value for the constant term
            stats_dict["t-stat"].append(t_stat)

            # Calculate the p-value from the t-statistic
            p_value = results.pvalues.iloc[0]  # p-value for the constant term
            stats_dict["p-value"].append(p_value)

        # Create a DataFrame from the statistics dictionary
        stats_df = pd.DataFrame(stats_dict, index=portfolio_columns)

        # Save the resulting DataFrame to a new Parquet file
        output_filepath = resultsPath + "/PortfoliosOutcomePlusDiffTest.parquet"
        stats_df.to_parquet(output_filepath, index=True)

        logger.info(f"Statistical test results saved to {output_filepath}")


# --------------------

# Create object and run all steps
if __name__ == "__main__":
    PortfolioSortsPipeline().runPipeline()