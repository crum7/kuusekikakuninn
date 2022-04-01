import re

s = 'aaa@xxx.com'

m = re.match(r'[a-z]+@[a-z]+\.[a-z]+', s)
print(m)