items = [("dog", 0.97), ("cat", 0.51)]

HIGH_LEVEL = 0.95
MED_LEVEL = 0.85


bins = {"high": [], "med": [], "low": []}

for item in items:
    level = item[-1]
    if level >= HIGH_LEVEL:
        bins["high"].append(item)
    elif level >= MED_LEVEL:
        bins["med"].append(item)
    else:
        bins["low"].append(item)

for key, value in bins.items():
    print(f"{key}: {len(value)}")
