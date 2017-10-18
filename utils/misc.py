""" This module contains some miscellaneous utility functions. """
import predictor.evaluation as ev
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial


NUM_PARTITIONS = 4  # number of partitions to split dataframe
NUM_CORES = 4  # number of cores on your machine


def unpack_params(params_df):
    """
    Helper function. Takes a one row parameters dataframe and returns its contents.
    :param params_df: A one row parameters dataframe
    :return: All the separated parameters.
    """
    GOOD_DATA_RATIO = params_df['GOOD_DATA_RATIO']
    train_val_time = int(params_df['train_val_time'])
    base_days = int(params_df['base_days'])
    step_days = int(params_df['step_days'])
    ahead_days = int(params_df['ahead_days'])
    SAMPLES_GOOD_DATA_RATIO = params_df['SAMPLES_GOOD_DATA_RATIO']
    x_filename = params_df['x_filename']
    y_filename = params_df['y_filename']
    return GOOD_DATA_RATIO, train_val_time, base_days, step_days, ahead_days, SAMPLES_GOOD_DATA_RATIO, x_filename, y_filename


def mean_score_eval(params, step_eval_days, eval_predictor):
    """
    Receives one set of parameters and returns the validation results of a rolling evaluation.
    """
    # Input values
    train_days = int(params['train_days'])

    GOOD_DATA_RATIO, \
        train_val_time, \
        base_days, \
        step_days, \
        ahead_days, \
        SAMPLES_GOOD_DATA_RATIO, \
        x_filename, \
        y_filename = unpack_params(params)

    pid = 'base{}_ahead{}_train{}'.format(base_days, ahead_days, train_days)
    print('Generating: {}'.format(pid))

    # Getting the data
    x = pd.read_pickle('../../data/{}'.format(x_filename))
    y = pd.read_pickle('../../data/{}'.format(y_filename))

    r2, mre, y_val_true_df, y_val_pred_df, mean_dates = ev.roll_evaluate(x,
                                                                         y,
                                                                         train_days,
                                                                         step_eval_days,
                                                                         ahead_days,
                                                                         eval_predictor,
                                                                         verbose=True)
    val_metrics_df = ev.get_metrics_df(y_val_true_df, y_val_pred_df)
    # return (pid, pid)
    result = tuple(val_metrics_df.mean().values)
    print(result)
    return result


def apply_mean_score_eval(params_df, step_eval_days, eval_predictor):
    """
    Helper function to use pandas apply with the mean score evaluator.
    """
    result_df = params_df.copy()
    result_df['scores'] = result_df.apply(partial(mean_score_eval,
                                                  step_eval_days=step_eval_days,
                                                  eval_predictor=eval_predictor), axis=1)
    return result_df


def dict_nan_to_none(d):
    result = {}
    for key in d.keys():
        if np.isnan(d[key]):
            result[key] = None
        else:
            result[key] = d[key]
    return result


def dict_float_to_int(d):
    result = {}
    for key in d.keys():
        if type(d[key]) is float:
            result[key] = int(d[key])
        else:
            result[key] = d[key]
    return result


def hyper_score_eval(hyper_df, params_df, step_eval_days, eval_predictor_class):
    hyper = dict_float_to_int(dict_nan_to_none(hyper_df.to_dict()))
    print('Evaluating: {}'.format(hyper))
    eval_predictor = eval_predictor_class()
    eval_predictor.set_params(**hyper)
    if hyper:
        return mean_score_eval(params_df, step_eval_days, eval_predictor)
    else:
        return None, None


def search_mean_score_eval(hyper_df, params_df, step_eval_days, eval_predictor_class):
    result_df = hyper_df.copy()
    result_df['scores'] = result_df.apply(partial(hyper_score_eval,
                                                  params_df=params_df,
                                                  step_eval_days=step_eval_days,
                                                  eval_predictor_class=eval_predictor_class),
                                          axis=1)
    return result_df


def parallelize_dataframe(params_df, func, params=None):
    """
    Takes a list of parameters in params_df, a function, and optionally some fixed
    params, and applies the function in parallel for all the parameters sets in df.
    :param params_df: A dataframe with one parameter set in each row.
    :param func: The function to apply.
    :param params: Some optional fixed parameters to pass to the function.
    :return: It returns the result of applying the function to each parameter set,
    in a dataframe.
    """
    df_split = np.array_split(params_df, NUM_PARTITIONS)
    pool = Pool(NUM_CORES)
    result_df = pd.concat(pool.map(partial(func, **params), df_split))
    pool.close()
    pool.join()
    return result_df
