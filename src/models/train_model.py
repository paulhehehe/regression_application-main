from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


# Function to train the model
def train_RFmodel(x, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)

    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')

    rfmodel = rf.fit(x_train,y_train)


    
    # Save the trained model
    with open('models/rfmodel.pkl', 'wb') as f:
        pickle.dump(rfmodel, f)

    return rfmodel, x_test, y_test