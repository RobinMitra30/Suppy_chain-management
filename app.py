from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


# Load the trained model
with open(r'D:\only projects\Machine learning projects\New folder\model_Dt.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from the form
    
    Location_type = request.form['Location_type']
    WH_capacity_size = request.form['WH_capacity_size']
    zone = request.form['zone']
    WH_regional_zone = request.form['WH_regional_zone']
    wh_owner_type = request.form['wh_owner_type']
    num_refill_req_l3m = int(request.form['num_refill_req_l3m'])
    transport_issue_l1y = int(request.form['transport_issue_l1y'])
    Competitor_in_mkt = int(request.form['Competitor_in_mkt'])
    electric_supply = int(request.form['electric_supply'])
    dist_from_hub = float(request.form['dist_from_hub'])
    workers_num = int(request.form['workers_num'])
    storage_issue_reported_l3m = int(request.form['storage_issue_reported_l3m'])
    temp_reg_mach = int(request.form['temp_reg_mach'])
    approved_wh_govt_certificate = int(request.form['approved_wh_govt_certificate'])
    wh_breakdown_l3m = int(request.form['wh_breakdown_l3m'])
    govt_check_l3m = int(request.form['govt_check_l3m'])
    

    # Create a dataframe with the input features
    input_data = pd.DataFrame({
        
        'Location_type_Urban': [1 if Location_type == 'Urban' else 0],
        'WH_capacity_size_Mid': [1 if WH_capacity_size == 'Small' else 0],
        'WH_capacity_size_Small': [1 if WH_capacity_size == 'Mid' else 0],
        'zone_North': [1 if zone == 'North' else 0],
        'zone_South': [1 if zone == 'South' else 0],
        'zone_West': [1 if zone == 'West' else 0],
        'WH_regional_zone_Zone 2': [1 if WH_regional_zone == 'Zone 2' else 0],
        'WH_regional_zone_Zone 3': [1 if WH_regional_zone == 'Zone 3' else 0],
        'WH_regional_zone_Zone 4': [1 if WH_regional_zone == 'Zone 4' else 0],
        'WH_regional_zone_Zone 5': [1 if WH_regional_zone == 'Zone 5' else 0],
        'WH_regional_zone_Zone 6': [1 if WH_regional_zone == 'Zone 6' else 0],
        'wh_owner_type_Rented': [1 if wh_owner_type == 'Rented' else 0],
        'num_refill_req_l3m': [num_refill_req_l3m],
        'transport_issue_l1y': [transport_issue_l1y ],
        'Competitor_in_mkt': [Competitor_in_mkt],
        'electric_supply': [electric_supply],
        'dist_from_hub' : [dist_from_hub],
        'workers_num': [workers_num],
        'storage_issue_reported_l3m': [storage_issue_reported_l3m],
        'temp_reg_mach': [temp_reg_mach],
        'approved_wh_govt_certificate' : [ approved_wh_govt_certificate ],
        'wh_breakdown_l3m': [wh_breakdown_l3m],
        'govt_check_l3m': [govt_check_l3m]
    })
    data_list = input_data.values.tolist()
    # Make prediction
    prediction = model.predict(np.array(data_list))

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)





