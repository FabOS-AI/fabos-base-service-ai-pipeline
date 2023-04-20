---
permalink: /docs/usage/pre-processing/
---

# Pre-Processing
This module contains the APIs in the file 'pp_api.py' which has calls to the pp_backend that has the logic to perform the pre-processing task required by the request.
This is hosted on the port '5000'.

## APIs
::: details Information
This api is called by sending the request to the main pre-processing path + '/info' and is a function that gives statistical information regarding the dataset.

function: 
    
    process_ts_info()

    It receives a data file when the function is called and reads it into a pandas dataframe.
    The dataframe is get all the statistical information regarding the dataset.
    
    :return: It returns a  json with information regarding the dataset
    :rtype: JSON
:::

:::details Remove NaN
This api is called by sending the request to Pre-processing path + '/nan' and is a function that removes all nan values.

function: 

    process_ts_rm_nan()
   
    It receives a data file when the function is called through the http request and reads it into the pandas dataframe.
    The dataframe is send to the :ref:`pp.rm_nan(df, column_list) <backend>` where all nan (not a number) elements are removed from all the columns or the selected target columns from the request.

    :return: It returns a dataframe converted to json with all the nan elements removed
    :rtype: JSON
:::

::: details Standardize
This api is called by sending the request to the main pre-processing path + '/std' and is a function that standardizes all values based on the target column defined in the request.

function: 

    process_ts_std()

    It receives a data file when the function is called and reads it into a pandas dataframe.
    The dataframe is send to the :ref:`pp.std(df, column_list) <backend>` as a function call to Standardize the values of the dataframe.

    :return: It returns a dataframe converted to json with standardized values
    :rtype: JSON
:::

::: details Normalization
This api is called by sending a http request to the main pre-processing path + '/norm' and is a function that normalizes all values based on the target column given in the request.

function: 

    process_ts_norm()

    It receives a data file when the function is called and reads it into a pandas dataframe.
    The dataframe is send to the :ref:`pp.norm(df, column_list) <backend>` as a function call to normalize all the values of the target columns.

    :return: It returns a dataframe converted to json with normalized values
    :rtype: JSON
:::

::: details Remove Co-relation
This api is called by sending a request to the main pre-processing path + '/rm_corr' and is a function that removes highly co-related features from the dataset based on the co-relation threshold.

function: 

    process_ts_corr()

    It receives a data file when the function is called and reads it into a pandas dataframe and if the threshold value is not present in the request it is set to '0.8'.
    The dataframe is send to the :ref:`pp.corr(df, corr_limit) <backend>` which is called as function to remove co-related features.

    :return: It returns a dataframe converted to json with strongly co-related features removed
    :rtype: JSON
:::