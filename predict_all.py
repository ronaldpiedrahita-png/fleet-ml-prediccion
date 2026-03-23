import requests

ok, errs = 0, 0
for truck_id in range(1, 201):
    try:
        r = requests.get(
            f"https://fleet-ml-prediccion-production.up.railway.app/predict/truck/{truck_id}/auto",
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if data["alert_level"] in ("CRITICAL", "HIGH"):
                print(f"  Camion {truck_id}: {data['alert_level']} prob={data['failure_prob']}")
            ok += 1
        else:
            errs += 1
    except Exception as e:
        errs += 1

print(f"\nCompletado: {ok} OK · {errs} errores")