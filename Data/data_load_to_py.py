import pandas as pd
import pickle as pk

df = pd.read_excel('Data/Single Sheet data.xlsx')
df.iloc[-1] = list(df.columns)
i = -2
output_data = []
answers = []
while True:
  i+=1
  try: row = df.iloc[i,:]
  except: break

  if row.iloc[0] != 'maze':
    answers = eval(row.iloc[1])
    constants = list(int(i) for i in answers.values())
    output_data.append([])
  else:
    maze_layout = eval(row.iloc[1])
    moves = eval(row.iloc[2].replace('true','True').replace('false','False'))
    output_data[-1].append((constants.copy(),maze_layout.copy(),moves.copy()))

with open('Data/saveddata.pk','wb') as f:
  pk.dump(output_data,f)