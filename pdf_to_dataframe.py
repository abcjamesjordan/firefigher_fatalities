import pandas as pd
import PyPDF2
import os

def append_series(series, df):
    series = pd.DataFrame(series).transpose()
    return pd.concat([df, series], ignore_index=True)

def get_ff(pageObj):
    page_text = pageObj.extractText()

    name = page_text.split('Fatality')[0]
    first_name = name.split()[0]
    last_name = name.split()[-1]
    middle_name = name.split()[1:-1]
    if isinstance(middle_name, list):
        for index, item in enumerate(middle_name):
            if '"' in item:
                middle_name.pop(index)
    middle_name = ''.join(middle_name)

    remaining = page_text.split('Fatality')[1]
    age = remaining.split('incidentinformation.')[1]
    age = age.split('Rank')[0]
    age = age.split('Age')[1]

    rank = remaining.split('Rank')[1]
    rank = rank.split('Classification')[0]

    classification = remaining.split('Classification')[1]
    classification = classification.split('Incident')[0]

    incident_date = remaining.split('Incident date')[1]
    incident_date = incident_date.split('Date of')[0]

    date_of_death = remaining.split('Date of death')[1]
    date_of_death = date_of_death.split('Cause of')[0]

    cause_of_death = remaining.split('Cause of death')[1]
    cause_of_death = cause_of_death.split('Nature of death')[0]

    nature_of_death = remaining.split('Nature of death')[1]
    nature_of_death = nature_of_death.split('Activity type')[0]

    activity = remaining.split('Activity type')[1]
    activity = activity.split('Emergency duty')[0]

    emergency = remaining.split('Emergency duty')[1]
    emergency = emergency.split('Duty type')[0]

    duty = remaining.split('Duty type')[1]
    duty = duty.split('Fixed property use')[0]

    property_type = remaining.split('Fixed property use')[1]
    property_type = property_type.split('Department information')[0]

    current_dict = {'first_name': first_name, 'last_name': last_name, 'middle_name': middle_name, 
                    'age': age, 'rank': rank, 'classification': classification, 
                    'incident_date': incident_date, 'date_of_death': date_of_death,
                    'cause_of_death': cause_of_death, 'nature_of_death': nature_of_death, 
                    'activity': activity, 'emergency': emergency, 'duty': duty, 'property_type': property_type}
    s = pd.Series(list(current_dict.values()), index=current_dict.keys())

    s['incident_date'] = pd.to_datetime(s['incident_date'])
    s['date_of_death'] = pd.to_datetime(s['date_of_death'])

    return s

# Paths
path = os.getcwd()
path_to_pdf_2021 = os.path.join(path, 'ffdata03292021.pdf')
path_to_csv = os.path.join(path, 'data/clean.csv')
path_to_pickle = os.path.join(path, 'data')

# Constants
cols_to_use = ['first_name', 'last_name', 'middle_name', 'age', 'rank', 'classification', 'incident_date', 'date_of_death', 'cause_of_death', 'nature_of_death', 'activity', 'emergency', 'duty', 'property_type',]

# creating a pdf file object
pdfFileObj = open(path_to_pdf_2021, 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# Create blank df
df_new = pd.DataFrame(columns=list(cols_to_use))

for page in range(pdfReader.numPages-1):
    pageObj = pdfReader.getPage(page)
    current_series = get_ff(pageObj)
    df_new = append_series(current_series, df_new)

df_new.to_pickle(os.path.join(path_to_pickle, 'df_2021'))
