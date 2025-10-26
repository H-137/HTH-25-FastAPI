import pandas as pd
import requests
from io import StringIO

def parse_divs_to_df(url):
    response = requests.get(url)
    response.raise_for_status()
    data_text = response.text
    
    #fixed width format requires some precise extraction, defined in the documentation for the website
    rows = []
    for line in data_text.strip().split('\n'):
        if len(line) < 94:
            continue
            
        row = {
            'state_code': line[0:2].strip(),
            'division': line[2:4].strip(),
            'element_code': line[4:6].strip(),
            'year': int(line[6:10].strip()),
            'jan': float(line[10:17].strip()),
            'feb': float(line[17:24].strip()),
            'mar': float(line[24:31].strip()),
            'apr': float(line[31:38].strip()),
            'may': float(line[38:45].strip()),
            'jun': float(line[45:52].strip()),
            'jul': float(line[52:59].strip()),
            'aug': float(line[59:66].strip()),
            'sep': float(line[66:73].strip()),
            'oct': float(line[73:80].strip()),
            'nov': float(line[80:87].strip()),
            'dec': float(line[87:94].strip())
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Map element codes, but we only need one which is avg temp
    element_map = {
        '02': 'Avg_Temperature',
    }
    df['element_name'] = df['element_code'].map(element_map)
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    df[month_cols] = df[month_cols].replace({  
        -99.90: pd.NA    
    })
    
    df['yearly_avg'] = df[month_cols].mean(axis=1, skipna=True)
    df = df[["state_code", "division", "element_code", "year", "yearly_avg"]]
    df = df[df['year']>1900]
    
    return df