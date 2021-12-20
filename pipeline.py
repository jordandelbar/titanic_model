from sklearn.Pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import (CategoricalImputer, 
                                       MeanMedianImputer)
from feature_engine.encoding import (MeanEncoder, 
                                     RareLabelEncoder)
from sklearn.ensemble import RandomForestClassifier

from titanic_model.config import config


titanic_pipeline = Pipeline([
        ('categorical_imputer', 
         CategoricalImputer(imputation_method='frequent', 
                            variables=config.model_config.cat_to_impute)),
        ('median_imputer', 
         MeanMedianImputer(imputation_method='median', 
                           variables=config.model_config.num_to_impute)),
        ('rare_label_encoder',
         RareLabelEncoder(tol=0.04,
                         variables=config.model_config.rare_label_to_group)),
        ('mean_target_encoder', 
         MeanEncoder(ignore_format=True,
                     variables=config.model_config.target_label_encoding)),
        ('scaling',
         ColumnTransformer([
            ('standard_scaler', StandardScaler(),
             config.model_config.feature_to_scale)
                ], remainder='passthrough')),
        ('clf',
         RandomForestClassifier())
    ]  
)