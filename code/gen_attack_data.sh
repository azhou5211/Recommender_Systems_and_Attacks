
# - Case 1: Low rating count and High rating target items 
# target items: [1122, 1201, 1500] selected items: [50, 181, 258]
python attack.py --ti 1122 1202 1500 --o ../attackData/case1

# - Case 2: Low rating count and Low rating target items
# target items: [1661, 1671, 1678] selected items: [50, 181, 258]
python attack.py --ti 1661 1671 1678 --o ../attackData/case2

# - Case 3: High rating count and Low rating target items
# target items: [678, 235, 210] selected items: [50, 181, 258]
python attack.py --ti  678 235 210 --o ../attackData/case3

# - Case 4: Random target items
# target items: [107, 62, 1216] selected items: [50, 181, 258]
python attack.py --ti 107 62 1216 --o ../attackData/case4

