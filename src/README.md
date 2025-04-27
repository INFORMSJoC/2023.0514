# Model Implementation

This section contains the code for our model implementation.

## Usage Instructions

1. Run the model with:
   ```python
   python main.py
   ```

2. Specify different transfer learning loss functions using the `--trans_loss_type` parameter:

  ```python
  python main.py --trans_loss_type mmd
  ```
This will run the model using MMD (Maximum Mean Discrepancy) as the transfer learning loss function

   