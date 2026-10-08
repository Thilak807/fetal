import re
with open('D:/fetal-health-classification-main/frontend/templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()
idx1 = html.find('id="tab-home"')
idx2 = html.find('id="tab-workstation"')
idx3 = html.find('Model Pipeline Overview')
print(f"tab-home at {idx1}")
print(f"tab-workstation at {idx2}")
print(f"Model Pipeline Overview at {idx3}")
