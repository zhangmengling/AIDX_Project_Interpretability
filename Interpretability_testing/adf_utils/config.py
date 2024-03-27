# -*- coding: utf-8 -*
import os

class pConfig:
    aPath = os.path.abspath('.')
    path = aPath[:aPath.index('project')] + 'project/'
    zip_path = path + "sav_demo/main/static/compressed/"
    data_path = path + "sav_demo/main/static/datasets/"
    model_path = path + "sav_demo/main/static/models/"
    explanation_path = path + "sav_demo/main/static/explanation/"
    tmp_path = path + "Interpretability_testing/"

class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])  #sensitive：gender
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    input_threshold = []
    input_threshold.append([[1, 2], [3, 4], [5, 6], [7, 8, 9]])
    input_threshold.append([[0, 1], [2, 3], [4, 5], [6, 7]])
    input_threshold.append([[0, 9], [10, 20], [21, 30], [31, 39]])
    input_threshold.append([[0, 3], [4, 7], [8, 11], [12, 15]])
    input_threshold.append([[0, 1], [2, 3], [4, 5], [6]])
    input_threshold.append([[0, 3], [4, 7], [8, 10], [11, 13]])
    input_threshold.append([[0, 1], [2, 3], [4], [5]])
    input_threshold.append([[0, 1], [2], [3], [4]])
    input_threshold.append([[0], [0], [1], [1]])
    input_threshold.append([[0, 45], [46, 99]])
    input_threshold.append([[0, 20], [21, 39]])
    input_threshold.append([[0, 45], [46, 99]])
    input_threshold.append([[0, 20], [21, 39]])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "√",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]
    target_names = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                                                                      "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]
    target_names = ['bad', 'good']

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]
    target_names = ['no', 'yes']

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class health_care:
    """
    Configuration of dataset Heart Failure Clinical Records
    Ouput: DEATH_EVENT (1: the patient deceased during the follow-up period; 0: otherwise)
    """

    # the size of total features
    params = 12

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([40, 95])
    input_bounds.append([0, 1])
    input_bounds.append([23, 7861])
    input_bounds.append([0, 1])
    input_bounds.append([14, 80])
    input_bounds.append([0, 1])
    input_bounds.append([25100, 850000])
    input_bounds.append([0.5, 9.4])
    input_bounds.append([113, 148])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([4, 285])

    # the name of each feature
    feature_name = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
                    "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

    # the name of each class
    class_name = ["no", "yes"]
    target_names = ['non_decrease', 'decrease']

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class diabetes_health:
    """
    Configuration of dataset Heart Failure Clinical Records
    Ouput: DEATH_EVENT (1: the patient deceased during the follow-up period; 0: otherwise)
    """

    # the size of total features
    params = 8

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 17])
    input_bounds.append([0, 199])
    input_bounds.append([0, 122])
    input_bounds.append([0, 99])
    input_bounds.append([0, 846])
    input_bounds.append([0, 67.1])
    input_bounds.append([0.078, 2.42])
    input_bounds.append([21, 81])

    # the name of each feature
    feature_name = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                    "PedigreeFunc", "Age"]

    # the name of each class
    class_name = ["no", "yes"]
    target_names = ['non', 'Diabetes']

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7]