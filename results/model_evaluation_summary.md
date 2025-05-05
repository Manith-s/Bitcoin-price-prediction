# Bitcoin Price Prediction: Model Evaluation Summary

Date: 2025-05-04 17:02:50

## Model Comparison

### Performance Metrics

| Model | RMSE | MAE | R² | Directional Accuracy | F1 Score |
| ----- | ---- | --- | -- | ------------------- | -------- |
| RandomForest | 13854.3137 | 7481.2648 | 0.6910 | 0.6879 | 0.6987 |
| GradientBoosting | 13748.7020 | 7379.1276 | 0.6957 | 0.7154 | 0.7155 |
| CatBoost | 22927.7369 | 15929.9871 | 0.1538 | 0.6055 | 0.6661 |
| LSTM | 0.7293 | 0.5047 | 0.4597 | 0.4831 | 0.4831 |
| GRU | 0.7502 | 0.5057 | 0.4284 | 0.4568 | 0.4561 |
| BidirectionalLSTM | 0.6594 | 0.4635 | 0.5584 | 0.4756 | 0.4747 |
| VotingEnsemble | 14496.4162 | 8039.9300 | 0.6617 | 0.7778 | 0.7778 |
| StackingEnsemble | 18308.7705 | 14297.6671 | 0.4604 | 0.3983 | 0.3983 |
| BlendingEnsemble | 14450.1663 | 7927.0660 | 0.6639 | 0.7266 | 0.7265 |

### Best Models

- Best model by RMSE: **BidirectionalLSTM** (RMSE: 0.6594)
- Best model by Directional Accuracy: **VotingEnsemble** (Accuracy: 0.7778)

## Feature Importance

The top 10 most important features across all models:

| Feature | Average Importance |
| ------- | ----------------- |
| high_rolling_max_30 | 0.0903 |
| low_rolling_median_14 | 0.0706 |
| low_lag_30 | 0.0640 |
| low | 0.0531 |
| high | 0.0528 |
| low_rolling_max_30 | 0.0509 |
| close_rolling_median_30 | 0.0395 |
| low_rolling_min_14 | 0.0349 |
| Bollinger_upper_50 | 0.0335 |
| low_rolling_mean_7 | 0.0298 |

## Conclusion and Recommendations

Based on the evaluation results, the following models performed best for Bitcoin price prediction:

1. **BidirectionalLSTM**
   - RMSE: 0.6594
   - Directional Accuracy: 0.4756
   - F1 Score: 0.4747

2. **LSTM**
   - RMSE: 0.7293
   - Directional Accuracy: 0.4831
   - F1 Score: 0.4831

3. **GRU**
   - RMSE: 0.7502
   - Directional Accuracy: 0.4568
   - F1 Score: 0.4561

### Recommendations

1. The **BidirectionalLSTM** model is recommended for production use based on overall performance.
2. Ensemble models generally performed better, suggesting that combining predictions from multiple models is beneficial.
3. Important features like technical indicators and price momentum should be monitored closely for trading decisions.
4. Models should be regularly retrained as new data becomes available to maintain accuracy.
