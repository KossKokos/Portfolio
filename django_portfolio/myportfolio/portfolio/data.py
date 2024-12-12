img_examples = [
    {
        'title': 'Recurent Neural Network',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/blstm_training_uitdwd.png"
    },
    {
        'title': 'Linear Regression',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/ridge_ibbdxz.png",
    },
    {
        'title': 'Clustering',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/clustering_aapydv.png",
    },
    {
        'title': 'SQL',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/sql_ntj4f4.png",
    },
    {
        'title': 'Deep Learning',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/comparing_models_oqgrh7.png",
    },
    {
        'title': 'Classification',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/comparing_results_kcnu15.png",
    },
    {
        'title': 'Data Analysis',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/lantitude_dz7phy.png",
    },
    {
        'title': 'Visualization',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/corr_matrix_jekicv.png"
    },
]


projects_data = [
    {
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733738120/car_plates_reader_cbgmgs.jpg',
        'title': 'Plates-Reader-Project',
        'introduction' : """The Car Parking application is a web-based app that automates parking management, 
                            including optical license plate recognition and parking duration tracking. 
                            It provides users with a convenient interface for viewing parking history 
                            and offers administrators tools to manage accounts and rates. 
                            The app simplifies the parking process and makes its management easier, 
                            ensuring convenience for all users.""",
        'git_url': 'https://github.com/KossKokos/Car-Parking-Project',
                'description': [
            {
                "heading": "User Management",
                "paragraph": """
                Fetch all users: Retrieve usernames of all users.
                Ban/Unban Users: Admins can ban or unban users by their IDs.
                Delete Users: Superadmins can delete user accounts except their own or another admin's without special permissions.
                Change Roles: Admins can update user roles, restricted by role hierarchy.
                                """,
                "code_snippet": """
@router.patch("/ban/{user_id}")
async def ban_user(user_id: str, ...):
    ...
    await repository_admin.update_banned_status(user, db)"""
                                },
            {
                "heading": "Car Management:", 
                "paragraph": """
                Ban/Unban Cars: Cars can be banned based on license plates, affecting user parking permissions.
                Search Users by Car: Locate users associated with a specific license plate.
                                """,
                "code_snippet": """
@router.patch("/ban_car/{license_plate}")
async def ban_car(license_plate: str, ...):
    ...
    await repository_cars.update_car_banned_status(car, db)"""
                                },
            {
                "heading": "CSV Export:", 
                "paragraph": """
                Generate a CSV file containing parking data for a specific car.
                                """,
                "code_snippet": """
@router.get("/create_csv/{license_plate}/{filename}")
async def create_csv_file(license_plate: str, filename: str, ...):
    ..."""
                                },
            {
                "heading": "Vehicle Detection (vehicle_detector.py)", 
                "paragraph": """
                For detecting vehicles, I used a YOLOv4 deep learning model. Here's what I did:
                I initialized the YOLOv4 network by loading its weights and configuration.
                The dnn module in OpenCV helps me integrate YOLO directly without setting up external dependencies.
                                """,
                "code_snippet": """
net = cv2.dnn.readNet(path + "/yolov4.weights", path + "/yolov4.cfg")
self.model = cv2.dnn_DetectionModel(net)
self.model.setInputParams(size=(832, 832), scale=1 / 255)"""
                                },
            {
                "heading": "Detecting Vehicles", 
                "paragraph": """
                When I pass an image to the model, it detects objects and filters them based on confidence and class:
                                """,
                "code_snippet": """
class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
if score >= 0.5 and class_id in self.classes_allowed:
    vehicles_boxes.append(box)"""
                                },
            {
                "heading": "Plate Reading (plate_reader.py)", 
                "paragraph": """
                After isolating vehicles, I focus on extracting and recognizing license plate text.
                I built a text classifier using Keras and trained it to recognize digits and letters on license plates:
                                """,
                "code_snippet": """
model = keras.models.load_model(path + r"/text_classifier.keras")"""
                                },
            {
                "heading": "Preprocessing for Text Detection", 
                "paragraph": """
                I developed a method to locate text-like regions in the image. This involves several steps:
                Convert the image to grayscale and enhance features.
                Extract contours and filter for size and aspect ratio.
                This gives me a set of candidate regions that might contain characters.
                                """,
                "code_snippet": """
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_thresh = cv2.adaptiveThreshold(img_blurred, 255.0, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 3)
if area > MIN_AREA and MIN_RATIO < ratio < MAX_RATIO:
    possible_contours.append(d)"""
                                },
            {
                "heading": "Cropping and Normalizing", 
                "paragraph": """
                Using the bounding boxes from the vehicle detector, I crop out the vehicle region.
                Then I resize and pad each character for consistent input to the recognition model.
                                """,
                "code_snippet": """
vehicle_boxes = self.vd.detect_vehicles(img)
x, y, w, h = vehicle_boxes[idx]
img = img[y:y+h, x:x+w]
img = cv2.resize(symbol, (15, 25))
img = await self.resize_with_pad(img, new_shape, color)"""
                                },
            {
                "heading": "Making Predictions", 
                "paragraph": """
                Finally, I pass the processed characters to my text classifier, 
                map predictions to their corresponding digits or letters, and assemble the plate text:
                                """,
                "code_snippet": """
predicted = self.model.predict(result)
result = [str(CLASSES[np.argmax(pred)]) for pred in predicted]
result = ''.join(result)"""
                                },
            {
                "heading": "How it All Fits Together", 
                "paragraph": """
                Detect Vehicles: I use YOLO to find vehicles in the input image.
                Crop the Vehicle: I isolate the largest bounding box, which likely contains the vehicle.
                Detect Plate Text: Using contour analysis, I identify character-like regions on the vehicle.
                Recognize Text: My Keras model predicts each character, reconstructing the license plate text.
                                """,
                "code_snippet": ''
                                },
        ]
    },
    {
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733740879/churn_predictions_gag4tv.png',
        'title': 'Churn-Prediction-Project',
        'introduction' : """The GitHub project titled "Customer_Churn_Prediction" focuses on 
                            predicting customer churn based on a dataset named Telco_customer_churn.csv. 
                            The repository contains a Jupyter Notebook (churn_predictions.ipynb) where analysis 
                            and modeling are implemented. I used data science techniques to identify 
                            patterns that predict whether a customer will leave a service.""",
        'git_url': 'https://github.com/KossKokos/Customer_Churn_Prediction',
        'description': [
            {
                "heading": "Getting Started",
                "paragraph": """
                First, I imported the essential libraries for data manipulation (pandas, numpy), 
                visualization (matplotlib), and machine learning tasks (scikit-learn). 
                I also suppressed warnings to keep the output clean.
                                """,
                "code_snippet": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')"""
                                },
            {
                "heading": "Loading and Exploring the Data", 
                "paragraph": """
                I started by loading the Telco Customer Churn dataset and displaying the 
                first five rows to understand its structure.
                Next, I checked the dataset’s size and structure using data.shape and data.info(). 
                This gave me an idea of the data types and any potential missing values.
                                """,
                "code_snippet": """
data = pd.read_csv('Telco_customer_churn.csv')
data.head()"""
                                },
            {
                "heading": "Cleaning the Data", 
                "paragraph": """
                While inspecting the data, I noticed the Count column was not useful for analysis. So, I dropped it.
                                """,
                "code_snippet": """
data.drop('Count', axis=1, inplace=True)"""
                                },
            {
                "heading": "Understanding Relationships", 
                "paragraph": """
                To find meaningful patterns, I analyzed numerical data using data.describe() 
                and visualized the correlation between features with a heatmap. 
                Strong correlations would indicate which features are impactful for predicting churn.
                                """,
                "code_snippet": """
data.corr(numeric_only=True)"""
                                },
            {
                "heading": "Churn Distribution Analysis", 
                "paragraph": """
                I wanted to visualize the distribution of churned vs. non-churned customers. 
                I created a bar chart to show this distribution, with colors differentiating between the two categories.
                                """,
                "code_snippet": """
count_yes = data[data['Churn Label'] == 'Yes'].count().values[0]
count_no = data['Churn Label'].count() - count_yes
fig, ax = plt.subplots()
ax.bar(labels, counts, color=colors)
plt.show()"""
                                },
            {
                "heading": "Data Preprocessing", 
                "paragraph": """
                I separated categorical and numerical columns, as they require different treatments. 
                For categorical columns, I used OneHotEncoder to convert them into numerical features.
                I then combined these encoded features with numerical data to form a complete dataset.
                                """,
                "code_snippet": """
from sklearn.preprocessing import OneHotEncoder
custom_fnames_enc = OneHotEncoder(sparse_output=False).fit(categorical_data)
categorical_vals = custom_fnames_enc.transform(categorical_data)
only_num_data = pd.concat([num_columns, transformed_data], axis=1)"""
                                },
            {
                "heading": "Model Development", 
                "paragraph": """
                I started with a simple logistic regression model. I scaled the features 
                using StandardScaler and split the data into training and testing sets. 
                Then, I trained the model and evaluated its performance using accuracy, precision, recall, and ROC-AUC metrics.
                I then explored Random Forest, tuning the number of estimators. 
                I found that 100 estimators performed best for this dataset.
                                """,
                "code_snippet": """
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)
y_pred_logistic = logistic_classifier.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=100)
randomforest_classifier.fit(X_train, y_train)"""
                                },
            {
                "heading": "Feature Importance", 
                "paragraph": """
                To understand which features were driving the predictions, I extracted the feature 
                importance values. This helped identify key drivers of churn, such as tenure or monthly_charges.
                                """,
                "code_snippet": """
feature_importance = logistic_classifier.coef_"""
                                },
            {
                "heading": "Advanced Modeling", 
                "paragraph": """
                I experimented with polynomial features to capture non-linear relationships 
                and tested models like SVM and SGDClassifier with grid searches for hyperparameter tuning.
""",
                "code_snippet": """
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 10, 100], 'solver': ['liblinear']}
grid_search = GridSearchCV(logistic_classifier, param_grid)
grid_search.fit(X_train, y_train)"""
                                },
            {
                "heading": "Visualization of Results", 
                "paragraph": """
                To validate the predictions visually, I plotted scatter plots comparing true and predicted churn values. 
                This allowed me to observe how well the model captured patterns in the data.
                                """,
                "code_snippet": """
plt.scatter(X_test_sample[4], X_test_sample[3], c=colors_list_pred)"""
                                },
        ]
    },
]


table_qualifications = [
    {
        'name': 'ESOL Level 1',
        'date_start': '03/09/2022',
        'date_finish': '31/07/2023',
        'grade': 'Completed'
    },
    
    {
        'name': 'Functional Skills Math Level 2',
        'date_start': '03/09/2022',
        'date_finish': '31/07/2023',
        'grade': 'Completed'
    },
    {
        'name': 'Functional Skills English Level 1',
        'date_start': '03/09/2022',
        'date_finish': '31/07/2023',
        'grade': 'Completed'
    },
    {
        'name': 'BTEC IT Level 1',
        'date_start': '03/09/2023',
        'date_finish': '31/07/2023',
        'grade': 'Completed'
    },
    {
        'name': 'GSCE Math Level 2',
        'date_start': '03/09/2023',
        'date_finish': '31/07/2024',
        'grade': '8'
    },
    {
        'name': 'GCSE English Level 2',
        'date_start': '03/09/2023',
        'date_finish': '31/07/2024',
        'grade': '4'
    },
    {
        'name': 'Python Developer GoIT',
        'date_start': '25/05/2023',
        'date_finish': '31/04/2024',
        'grade': 'Completed'
    },
        {
        'name': 'BTEC IT Level 2',
        'date_start': '03/09/2024',
        'date_finish': 'Current',
        'grade': 'In Progress'
    },
]


services_data = [
    {
        'title': 'Regression:',
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/ridge_ibbdxz.png',
        'description': 
                    """I have extensive experience building predictive models using various regression 
                    techniques, such as linear, polynomial, and regularized regression (Ridge, Lasso). 
                    These models are effective in predicting continuous outcomes, whether it’s 
                    forecasting sales, predicting stock prices, or estimating real-world metrics 
                    from historical data. By analyzing data trends and applying statistical methods, 
                    I can create models that not only fit the data well but are also robust to overfitting, 
                    ensuring high generalization performance.""" 
    },

    {
        'title': 'Classification:',
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/comparing_results_kcnu15.png',
        'description': 
                    """I have worked with a variety of classification algorithms, 
                    such as Logistic Regression, Decision Trees, Random Forests, 
                    Support Vector Machines (SVM), and Naive Bayes to solve problems 
                    with categorical outcomes. From binary classification (like spam detection 
                    or customer churn) to multi-class classification (such as classifying different 
                    types of products or diseases), I understand how to select the most appropriate
                    model, preprocess data, and evaluate model performance using metrics like 
                    accuracy, precision, recall, and F1-score.""" 
    },
    
    {
        'title': 'Clustering:',
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/clustering_aapydv.png',
        'description': 
                    """With a focus on unsupervised learning, I have applied clustering 
                    algorithms like K-Means, DBSCAN, and Hierarchical Clustering to identify 
                    patterns or groupings in data without labeled outcomes. This has been 
                    particularly useful for segmenting customer data for targeted marketing 
                    campaigns, grouping similar products for recommendation systems, or 
                    identifying outliers in large datasets. I am also adept at visualizing 
                    clusters and using dimensionality reduction techniques to make the results 
                    more interpretable.""" 
    },
    {
        'title': 'Deep Learning:',
        'url': 'https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/comparing_models_oqgrh7.png',
        'description': 
                    """I have hands-on experience in designing, training, and deploying 
                    neural networks for a wide range of applications. I specialize 
                    in Convolutional Neural Networks (CNNs) for image-related tasks 
                    like image classification, object detection, and image segmentation. 
                    I also have experience working with Recurrent Neural Networks 
                    (RNNs), including LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units),
                    for time-series data and sequential problems, such as predicting stock 
                    prices or generating text. By utilizing frameworks like TensorFlow and 
                    Keras, I am capable of creating complex, deep models to solve cutting-edge problems.
""" 
    },
]