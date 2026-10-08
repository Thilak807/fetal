import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import create_app

app = create_app()
client = app.test_client()

# 1. Test demo patients list
res = client.get('/api/demo/patients')
assert res.status_code == 200
data = res.get_json()
print(f"Retrieved {len(data)} synthetic patients from API.")
assert len(data) == 50

# 2. Test loading patient 1001
res2 = client.post('/api/demo/load_patient/PAT-1001')
assert res2.status_code == 200
data2 = res2.get_json()
p_name = data2['patient']['name']
p_id = data2['patient']['patient_id']
print(f"Successfully loaded {p_id} ({p_name})")
assert 'brain' in data2 and 'cardiac' in data2

# 3. Test multi-modal fusion prediction
res3 = client.post('/api/predict/multimodal', json={
    'patient_id': data2['patient']['id'],
    'brain_image_id': data2['brain']['db_id'],
    'cardiac_signal_id': data2['cardiac']['db_id'],
    'clinician_notes': 'Verified test evaluation'
})
assert res3.status_code == 200
data3 = res3.get_json()
pred = data3['prediction']
conf = data3['confidence']
risk = data3['risk_percent']
print(f"Multi-modal prediction: {pred} (Confidence: {conf}%, Risk: {risk}%)")

# 4. Test loading suspect case 1035
res_s = client.post('/api/demo/load_patient/PAT-1035')
assert res_s.status_code == 200
data_s = res_s.get_json()
res_sp = client.post('/api/predict/multimodal', json={
    'patient_id': data_s['patient']['id'],
    'brain_image_id': data_s['brain']['db_id'],
    'cardiac_signal_id': data_s['cardiac']['db_id'],
})
data_sp = res_sp.get_json()
print(f"Suspect case PAT-1035 prediction: {data_sp['prediction']} (Risk: {data_sp['risk_percent']}%)")

print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
