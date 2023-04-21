import requests
import pandas as pd
import numpy as np
import logging

from flask import jsonify

import tests.preprocessing_tests as pp
import tests.test_data as td
# import tests.ai_toolbox_tests as tbox
import tests.workflow_tests as wft
import tests.use_case_tests as uct

#logging.basicConfig(filename='BaseServiceInfo.LOG', encoding='utf-8', format='%(asctime)s #### %(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    ### Run differet base service test sets.

    # Run preprocessing tests
    # my_pp_tests = pp.PreprocessingTests()
    # my_pp_tests.run_time_series_tests()

    # Run toolbox tests
    # tobx_tests = tbox.AI_ToolboxTests()
    # tbox.run_tests(ts=True,img= False)    
    
  

    # TODO die tests schreiben
    # 1. Dataframe preprocessing
    # 2. AI-Toolbox Modelling
    # 3. xAI-Toolbox
    # 4. Evaluatin Service
    # 5. Pipeline computation
    # 6. IPA <-> IPT Extraction Pipeline

    uc_tests = uct.UseCaseTests()
    uc_tests.dataframe_preprocessing()
    # uc_tests.ai_toolbox_modelling()
    # uc_tests.xai_service_tests()
    # uc_tests.evaluation_service_single_model()
    # uc_tests.evaluation_service_model_benchmark()




    # TODO 
    # - AI Toolbox testen
    # Run xAI tests
    # Run Evaluation/Benchmarking tests
    # Run Transfer Learning tests
    # Run Workflow tests
    # wf_tests = wft.WorkflowTests()
    # wf_tests.run_pipeline()