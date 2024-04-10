import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the data
data1 = {
    'Method/Model': ['Luhn Algorithm', 'TextRank Algorithm', 'Keyword-based method', 'Combined-Extractive method'],
    'ROUGE-1 Scores': [0.190298, 0.180377, 0.169949, 0.171206],
    'ROUGE-2 Scores': [0.093097, 0.086007, 0.086664, 0.088783],
    'ROUGE-L Scores': [0.131865, 0.125163, 0.119708, 0.122091],
    'ROUGE-Lsum Scores': [0.161282, 0.152629, 0.144732, 0.147022]
}
data2 = {
    'Method/Model': ['Luhn Algorithm', 'TextRank Algorithm', 'Keyword-based method', 'Combined-Extractive method'],
    'ROUGE-1 Scores': [0.255721, 0.230912, 0.226269, 0.226259],
    'ROUGE-2 Scores': [0.107668, 0.087918, 0.097719, 0.100074],
    'ROUGE-L Scores': [0.16872, 0.15325, 0.148164, 0.151871],
    'ROUGE-Lsum Scores': [0.212751, 0.193782, 0.185815, 0.190122]
}
data3 = {
    'Method/Model': ['BART Model', 'T5 Model', 'PEGASUS Model'],
    'ROUGE-1 Scores': [0.416393, 0.327469, 0.482442],
    'ROUGE-2 Scores': [0.199679, 0.139161, 0.28407],
    'ROUGE-L Scores': [0.301758, 0.231178, 0.378892],
    'ROUGE-Lsum Scores': [0.360194, 0.285427, 0.436047]
}
df = pd.DataFrame(data3)

# Set the 'Method/Model' column as the index
df.set_index('Method/Model', inplace=True)

# Plot the data
df.plot(kind='bar', figsize=(8, 4))
plt.title('ROUGE Scores for Different Pretrained models')
plt.xlabel('Method/Model')
plt.ylabel('ROUGE Score')
# plt.ylim(0, 0.5) 
plt.xticks(rotation=360)  # Rotate x-axis labels for better readability
# plt.legend(title='ROUGE Score Type')
plt.legend(title='ROUGE Score Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
