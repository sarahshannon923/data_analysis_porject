import pandas as pd 

#import dataset
sd_2024 = pd.read_csv("vgchartz-2024-2.csv")
sd_2024.head(10)

#data cleaning
clean_sd_2024 = sd_2024.dropna()
print(clean_sd_2024)

#visualize in power bi

clean_sd_2024.to_csv("sales_insigt_data.csv", index=False)
