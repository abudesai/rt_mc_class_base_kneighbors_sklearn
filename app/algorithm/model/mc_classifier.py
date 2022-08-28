
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from sklearn.neighbors import KNeighborsClassifier


model_fname = "model.save"
MODEL_NAME = "multi_class_base_kneighbors_sklearn"


class Classifier(): 
    
    def __init__(self, n_neighbors = 5, weights = "uniform", p = 2, algorithm = "auto", leaf_size = 30, **kwargs) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        
        
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = KNeighborsClassifier(n_neighbors = self.n_neighbors, weights = self.weights, p = self.p, algorithm = self.algorithm, leaf_size = self.leaf_size)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X, verbose=False): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        knnclassifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return knnclassifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


