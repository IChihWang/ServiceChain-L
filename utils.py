import logging
import os
import numpy as np

def configure_logging(path_to_log_directory):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    #handler = logging.FileHandler('10p_AACRLCONV_log_{:s}.log'.format(hparam))
    #handler = logging.FileHandler('arewm_expl_mwm_ddpg.log')
    filename=os.path.join(path_to_log_directory, "result.log")
    #print("sdfsdfdsfdsfdsf:", filename)

    handler = logging.FileHandler(filename=os.path.join(path_to_log_directory, "result.log"))
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    #logger.info(hparam)
    return logger

logger = configure_logging("./")