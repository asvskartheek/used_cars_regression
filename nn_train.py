import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class CarPriceDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for car price prediction.

    This module handles data loading, preprocessing, and preparation for training.

    Args:
        data_dir (str): Directory containing the data files.
        batch_size (int): Batch size for data loaders.
        min_category_count (int): Minimum count for a category to be considered.

    Attributes:
        CATS (list): List of categorical feature names.
        NUMS (list): List of numerical feature names.
        cat_feature_sizes (list): List of sizes for each categorical feature.
        cat_embedding_sizes (list): List of embedding sizes for each categorical feature.
        train (pd.DataFrame): Preprocessed training data.
        test (pd.DataFrame): Preprocessed test data.
    """

    def __init__(self, data_dir: str, batch_size: int, min_category_count: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.min_category_count = min_category_count
        self.CATS = None
        self.NUMS = None
        self.cat_feature_sizes = None
        self.cat_embedding_sizes = None

    def prepare_data(self):
        """
        Prepare the data if needed.

        This method is called only on 1 GPU in distributed settings.
        """
        pass

    def setup(self, stage: str = None):
        """
        Read, preprocess, and set up the data.

        This method handles data loading, feature engineering, and preprocessing.

        Args:
            stage (str, optional): Stage of setup (fit or test). Defaults to None.
        """
        # Read and preprocess the data
        train = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        test = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        test["price"] = 0  # For common pre-processing steps

        combined = pd.concat([train, test], axis=0, ignore_index=True)

        self.NUMS = ["milage"]
        self.CATS = [c for c in combined.columns if c not in ["id", "price"] + self.NUMS]

        # Standardize numerical features
        for c in self.NUMS:
            m, s = combined[c].mean(), combined[c].std()
            combined[c] = (combined[c] - m) / s
            combined[c] = combined[c].fillna(0)

        # Label encode categorical features
        self.cat_feature_sizes = []
        self.cat_embedding_sizes = []
        for cat_feature in self.CATS:
            combined[cat_feature], _ = combined[cat_feature].factorize()
            combined[cat_feature] -= combined[cat_feature].min()
            value_counts = combined[cat_feature].value_counts()

            rare_categories = value_counts[value_counts < self.min_category_count].index
            max_freq = combined[cat_feature].max()

            self.cat_feature_sizes.append(max_freq + 2)
            self.cat_embedding_sizes.append(int(np.ceil(np.sqrt(max_freq + 2))))

            combined[cat_feature] += 1
            combined.loc[combined[cat_feature].isin(rare_categories), cat_feature] = 0

        self.train = combined[:len(train)]
        self.test = combined[len(train):]

    def train_dataloader(self):
        """
        Create and return the training DataLoader.

        Note: This method is not used for K-Fold CV. Dataloaders are created in the training loop instead.

        Returns:
            DataLoader: DataLoader for the training data.
        """
        X_train_cats = torch.tensor(self.train[self.CATS].values, dtype=torch.long)
        X_train_nums = torch.tensor(self.train[self.NUMS].values, dtype=torch.float32)
        y_train = torch.tensor(self.train["price"].values, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_cats, X_train_nums, y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Create and return the validation DataLoader.

        Note: This method is not used for K-Fold CV. Dataloaders are created in the training loop instead.

        Returns:
            DataLoader: DataLoader for the validation data (same as training data in this case).
        """
        return self.train_dataloader()

    def test_dataloader(self):
        """
        Create and return the test DataLoader.

        Returns:
            DataLoader: DataLoader for the test data.
        """
        X_test_cats = torch.tensor(self.test[self.CATS].values, dtype=torch.long)
        X_test_nums = torch.tensor(self.test[self.NUMS].values, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_cats, X_test_nums)
        return DataLoader(test_dataset, batch_size=self.batch_size)


class CarPriceModel(pl.LightningModule):
    """
    PyTorch Lightning module for car price prediction.

    This module defines the neural network architecture and training process.

    Args:
        cat_feature_sizes (list): List of sizes for each categorical feature.
        cat_embedding_sizes (list): List of embedding sizes for each categorical feature.
        num_cat_features (int): Number of categorical features.
        num_num_features (int): Number of numerical features.
    """

    def __init__(
        self, cat_feature_sizes, cat_embedding_sizes, num_cat_features, num_num_features
    ):
        super(CarPriceModel, self).__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_feature_sizes[i], cat_embedding_sizes[i])
                for i in range(num_cat_features)
            ]
        )

        total_embedding_dim = sum(cat_embedding_sizes)
        self.fc1 = nn.Linear(total_embedding_dim + num_num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss()

    def forward(self, cat_inputs, num_inputs):
        """
        Forward pass of the model.

        Args:
            cat_inputs (torch.Tensor): Categorical input features.
            num_inputs (torch.Tensor): Numerical input features.

        Returns:
            torch.Tensor: Model predictions.
        """
        embedded = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([embedded, num_inputs], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Tuple containing cat_inputs, num_inputs, and target.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        cat_inputs, num_inputs, y = batch
        y_hat = self(cat_inputs, num_inputs)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Tuple containing cat_inputs, num_inputs, and target.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        cat_inputs, num_inputs, y = batch
        y_hat = self(cat_inputs, num_inputs)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log("val_loss", loss)
        return loss

    def lr_lambda(self, epoch):
        """
        Learning rate scheduler function.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Learning rate for the given epoch.
        """
        lr_schedule = [0.001] * 2 + [0.0001] * 1
        # Add a condition to handle epochs beyond the lr_schedule length
        # This is necessary because PyTorch Lightning sometimes calls for epoch 3 unexpectedly
        return lr_schedule[epoch] if epoch < len(lr_schedule) else lr_schedule[-1]

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            tuple: Tuple containing a list of optimizers and a list of schedulers.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return [optimizer], [scheduler]


def train_model(
    datamodule,
    num_folds,
    random_state,
    epochs,
    train_batch_size,
    eval_batch_size,
    num_workers,
    model_checkpoint_dir,
):
    """
    Train the model using k-fold cross-validation.

    Args:
        datamodule (CarPriceDataModule): The data module containing the dataset.
        num_folds (int): Number of folds for cross-validation.
        random_state (int): Random state for reproducibility.
        epochs (int): Number of training epochs.
        train_batch_size (int): Batch size for training.
        eval_batch_size (int): Batch size for evaluation.
        num_workers (int): Number of workers for data loading.
        model_checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
        tuple: Out-of-fold predictions and test predictions.
    """
    print("Starting model training...")
    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    out_of_fold_predictions = np.zeros(len(datamodule.train))
    test_predictions = np.zeros(len(datamodule.test))

    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    for fold, (train_index, val_index) in enumerate(kf.split(datamodule.train)):
        print(f"{'#' * 25}\n### Fold {fold} ###\n{'#' * 25}")

        train_dataset = TensorDataset(
            torch.tensor(datamodule.train.iloc[train_index][datamodule.CATS].values, dtype=torch.long, device=device),
            torch.tensor(datamodule.train.iloc[train_index][datamodule.NUMS].values, dtype=torch.float32, device=device),
            torch.tensor(datamodule.train.iloc[train_index]["price"].values, dtype=torch.float32, device=device)
        )

        val_dataset = TensorDataset(
            torch.tensor(datamodule.train.iloc[val_index][datamodule.CATS].values, dtype=torch.long, device=device),
            torch.tensor(datamodule.train.iloc[val_index][datamodule.NUMS].values, dtype=torch.float32, device=device),
            torch.tensor(datamodule.train.iloc[val_index]["price"].values, dtype=torch.float32, device=device)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        model = CarPriceModel(
            datamodule.cat_feature_sizes,
            datamodule.cat_embedding_sizes,
            len(datamodule.CATS),
            len(datamodule.NUMS)
        ).to(device)

        wandb_logger = WandbLogger(project="car_price_prediction", name=f"fold_{fold}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_checkpoint_dir,
            filename=f"best_model_fold_{fold}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        class MetricsCallback(pl.Callback):
            """
            Callback for logging metrics during training and validation.
            """

            def on_train_epoch_end(self, trainer, pl_module):
                """
                Log training RMSE at the end of each training epoch.

                Args:
                    trainer (pl.Trainer): PyTorch Lightning trainer instance.
                    pl_module (pl.LightningModule): PyTorch Lightning module instance.
                """
                train_rmse = torch.sqrt(trainer.callback_metrics["train_loss"])
                trainer.logger.log_metrics(
                    {f"train_rmse_fold_{fold}": train_rmse}, step=trainer.global_step
                )

            def on_validation_epoch_end(self, trainer, pl_module):
                """
                Log validation RMSE at the end of each validation epoch.

                Args:
                    trainer (pl.Trainer): PyTorch Lightning trainer instance.
                    pl_module (pl.LightningModule): PyTorch Lightning module instance.
                """
                val_rmse = torch.sqrt(trainer.callback_metrics["val_loss"])
                trainer.logger.log_metrics(
                    {f"val_rmse_fold_{fold}": val_rmse}, step=trainer.global_step
                )

        metrics_callback = MetricsCallback()

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback, metrics_callback],
            log_every_n_steps=50,
            deterministic=True,
            accelerator=device.type,
        )

        trainer.fit(model, train_loader, val_loader)

        # Load best model for predictions
        best_model = CarPriceModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            cat_feature_sizes=datamodule.cat_feature_sizes,
            cat_embedding_sizes=datamodule.cat_embedding_sizes,
            num_cat_features=len(datamodule.CATS),
            num_num_features=len(datamodule.NUMS),
        ).to(device)

        best_model.eval()
        with torch.no_grad():
            val_predictions = []
            for batch in val_loader:
                cat_features, num_features, _ = batch # _ is the target
                batch_predictions = best_model(cat_features, num_features)
                val_predictions.extend(batch_predictions.cpu().numpy().flatten())
            val_predictions = np.array(val_predictions)

            fold_test_predictions = []
            for batch in datamodule.test_dataloader():
                cat_features, num_features = [b.to(device) for b in batch]
                batch_predictions = best_model(cat_features, num_features)
                fold_test_predictions.extend(
                    batch_predictions.detach().cpu().numpy().flatten()
                )
            fold_test_predictions = np.array(fold_test_predictions)

        out_of_fold_predictions[val_index] = val_predictions
        test_predictions += fold_test_predictions / num_folds

        rmse = np.sqrt(
            np.mean(
                (val_predictions - datamodule.train.iloc[val_index]["price"].values)
                ** 2
            )
        )
        print(f" => Validation RMSE = {rmse:.4f}\n")

        # Log fold results to wandb
        wandb.log(
            {
                f"fold_{fold}_val_rmse": rmse,
            }
        )

    print("Model training completed.")
    return out_of_fold_predictions, test_predictions


def save_predictions(
    datamodule,
    out_of_fold_predictions,
    test_predictions,
    version,
    predictions_dir,
    data_dir,
):
    """
    Save out-of-fold and test predictions to CSV files and log to wandb.

    Args:
        datamodule (CarPriceDataModule): The data module containing the dataset
        out_of_fold_predictions (np.array): Out-of-fold predictions
        test_predictions (np.array): Test predictions
        version (int): Version number
        predictions_dir (str): Directory to save predictions
        data_dir (str): Directory containing the data
    """
    print("Saving predictions...")
    oof_df = datamodule.train[["id"]].copy()
    oof_df["pred"] = out_of_fold_predictions
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    oof_path = os.path.join(predictions_dir, f"oof_v{version}_{timestamp}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"Out-of-fold predictions saved to {oof_path}")

    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    sub.price = test_predictions
    sub_path = os.path.join(predictions_dir, f"submission_v{version}_{timestamp}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Test predictions saved to {sub_path}")

    # Log predictions to wandb
    wandb.log(
        {
            "oof_predictions": wandb.Table(dataframe=oof_df),
            "test_predictions": wandb.Table(dataframe=sub),
        }
    )


if __name__ == "__main__":
    # Constants
    VERSION = 1
    NUM_FOLDS = 5
    RANDOM_STATE = 42
    EPOCHS = 10
    MIN_CATEGORY_COUNT = 40
    TRAIN_BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 512
    NUM_WORKERS = 1

    # Paths
    DATA_DIR = "data/"
    MODEL_CHECKPOINT_DIR = "checkpoints/"
    PREDICTIONS_DIR = "predictions/"

    print("PyTorch Version", torch.__version__)
    print("PyTorch Lightning Version", pl.__version__)

    # Initialize wandb
    wandb.init(
        project="car_price_prediction",
        config={
            "version": VERSION,
            "num_folds": NUM_FOLDS,
            "random_state": RANDOM_STATE,
            "epochs": EPOCHS,
            "min_category_count": MIN_CATEGORY_COUNT,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
    )
    print("Starting main process...")
    # Create necessary directories
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    print(f"Created directories: {MODEL_CHECKPOINT_DIR}, {PREDICTIONS_DIR}")

    # Initialize and setup the data module
    datamodule = CarPriceDataModule(DATA_DIR, TRAIN_BATCH_SIZE, MIN_CATEGORY_COUNT)
    datamodule.setup()

    # Train model and get predictions
    out_of_fold_predictions, test_predictions = train_model(
        datamodule,
        NUM_FOLDS,
        RANDOM_STATE,
        EPOCHS,
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        NUM_WORKERS,
        MODEL_CHECKPOINT_DIR,
    )

    # Compute and display CV RMSE score
    rmse = np.sqrt(
        np.mean((out_of_fold_predictions - datamodule.train.price.values) ** 2)
    )
    print(f"Overall CV RMSE = {rmse:.4f}")

    # Log final RMSE to wandb
    wandb.log({"cv_rmse": rmse})

    # Save predictions
    save_predictions(
        datamodule,
        out_of_fold_predictions,
        test_predictions,
        VERSION,
        PREDICTIONS_DIR,
        DATA_DIR,
    )

    print("Main process completed.")
    wandb.finish()
