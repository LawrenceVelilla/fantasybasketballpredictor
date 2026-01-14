from inference import evaluate_player
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = evaluate_player(
      player_name="keyonte george",
      test_season=2025
)

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['actual_fpts'], label='Actual Fantasy Pts', marker='o')
plt.plot(df['date'], df['predicted_fpts'], label='Predicted Fantasy Pts', marker='x')
plt.xticks(rotation=45)
plt.xlabel('Game Date')
plt.ylabel('Fantasy Points')
plt.title('Actual vs Predicted Fantasy Points for Shai Gilgeous-Alexander (2023-24 Season)')
plt.legend()
plt.tight_layout()
plt.show()  
