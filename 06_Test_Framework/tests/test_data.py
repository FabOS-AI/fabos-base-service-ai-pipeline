import os
import base64

from sklearn.datasets import make_moons, make_circles, make_classification, make_regression

nr_samples = 500

def make_moons_ds():

    return make_moons(n_samples=nr_samples, shuffle=True, random_state=42)
    pass


def make_circles_ds():

    return make_circles(n_samples=nr_samples, noise=0.2, factor=0.5, random_state=42)
    pass


def make_classification_ds():

    return make_classification(n_samples=nr_samples, n_features = 4, n_clusters_per_class=1)
    pass

# Regression datasets
def make_regression_ds():

    return make_regression(n_samples=nr_samples)
    pass


def create_image_test_data():
    """
    Create a test dataset which contains tuples of images and classifications
    """

    # Create training data as combination of base64 string  + class ID
    DIR = "../06_Daten/Magnetic-tile-defect-datasets-master"
    CATEGORIES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free", "MT_Uneven"]
    IMG_SIZE = 128
    training_data = []

    b64_ims = []
    class_ids = []

    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = CATEGORIES.index(category)
    
        for img in os.listdir(path):

            filename = os.path.join(path, img)

            with open(filename, 'rb') as f:
                im_b64 = base64.b64encode(f.read())
                
                # Add tuple of image and category
                b64_ims.append(im_b64)
                class_ids.append(category)
                pass        
            
            pass
        
        pass
    
    return b64_ims, class_ids

    pass