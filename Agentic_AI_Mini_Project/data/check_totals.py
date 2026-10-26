import csv

rows = list(csv.DictReader(open('data/mock_expenses.csv')))

RULES = [
    (['flight','airline','airways','delta','united','southwest'], 'Flights'),
    (['hotel','inn','hilton','marriott','holiday'], 'Hotels'),
    (['grille','grill','cafe','starbucks','coffee','subway','mcdonald','cheesecake','olive garden', 'restaurant'], 'Meals'),
    (['uber','lyft','taxi','bart','metro','enterprise','amtrak','parking'], 'Transport'),
]

def infer(vendor, desc):
    text = f'{vendor} {desc}'.lower()
    for kws, cat in RULES:
        if any(k in text for k in kws):
            return cat
    return 'Misc'

total_by_cat = {}
missing = []
for r in rows:
    amt = r['amount'].strip()
    cat = r['category'].strip() or infer(r['vendor'], r['description'])
    if not amt or float(amt) == 0.0:
        missing.append(f"{r['vendor']} ({cat})")
        continue
    total_by_cat[cat] = total_by_cat.get(cat, 0) + float(amt)

print('Simulated category totals:')
for c,v in sorted(total_by_cat.items()):
    print(f'  {c}: {v:.2f}')
print(f'Grand total: {sum(total_by_cat.values()):.2f}')
print(f'Missing/zero rows: {missing}')
