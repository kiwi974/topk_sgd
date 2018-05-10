import sparseToolsDict as std

d1 = {1:1,2:8}
d2 = {1:3}
d3 = {2:1}
d4 = {3:2}
d5 = {2:6}

print(std.merge([d1,d2,d3,d4,d5],5))