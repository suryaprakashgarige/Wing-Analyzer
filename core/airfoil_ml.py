import joblib
import numpy as np
import os

class AirfoilML:
    def __init__(self, models_dir):
        """
        Initializes the ML loader with support for Cl and Cd models.
        """
        cl_path = os.path.join(models_dir, 'model_cl.pkl')
        cd_path = os.path.join(models_dir, 'model_cd.pkl')
        
        if not os.path.exists(cl_path) or not os.path.exists(cd_path):
            raise FileNotFoundError(f"Model files not found in {models_dir}")
            
        self.model_cl = joblib.load(cl_path)
        self.model_cd = joblib.load(cd_path)
        print(f"Models loaded successfully from {models_dir}")

    def predict(self, re, aoa, thickness, thickness_loc, camber, camber_loc):
        """
        Predicts Cl and Cd for a given set of airfoil features.
        
        Note: The repository models expect exactly 6 features.
        Features vector order: [Re, alpha, thickness, thickness_loc, camber, camber_loc]
        """
        x = np.array([[re, aoa, thickness, thickness_loc, camber, camber_loc]])
        
        cl = float(self.model_cl.predict(x)[0])
        cd = float(self.model_cd.predict(x)[0])
        
        return cl, cd
