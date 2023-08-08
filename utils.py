def categorize_age(age):
    if age < 30: 
        cat = '<30'
    elif age < 50:
        cat = '30-49'
    elif age < 70:
        cat = '50-69'
    elif age < 90:
        cat = '70-89'
    else: 
        cat = '>90'
    return cat