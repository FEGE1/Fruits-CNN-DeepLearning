from datetime import datetime

date = datetime.today()
date = str(date.day)+"-"+str(date.month)+"-"+str(date.year)+"-"+str(date.hour)+":"+str(date.minute)+":"+str(date.second)

print(date)