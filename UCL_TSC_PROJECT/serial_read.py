import serial
import pandas as pd

ser = serial.Serial('com5', 9600)
print("Accessing complete.")
store = []
count = 0

#while True:
while count < 10:
    print("Readling {} begin".format(count))
    data = ser.readline()
    data = data.decode() # converte bytes into string
    #data = data.split(" ")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
    #data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
    print(data)
    store.append(data)  # 添加到列表里
    count = count + 1

print("Writting begin.")
df = pd.DataFrame(store)  # 转化为df格式数据
df.to_excel('C:/Users/ASUS/Desktop/serial.xlsx', header=False, index=False)
print("Writting complete.")