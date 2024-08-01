import numpy as np
import torch
from pytorch_tabnet.utils import filter_weights
from pytorch_tabnet.abstract_model import TabModel
import scipy
from scipy.sparse import csc_matrix
from pytorch_tabnet.utils import create_sampler, check_warm_start
from pytorch_tabnet.stock_dataloader_cs import (
    StockDatasetCS,
    StockDataLoaderCS,
    StockPredictDatasetCS,
    StockPredictDataLoaderCS,
    validate_eval_set_stocks,
)
import warnings


class TabNetStockRegressor(TabModel):
    def __post_init__(self):
        super().__post_init__()
        self._task = "regression"
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = "mse"

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        self.output_dim = 1
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score

    def _construct_loaders(self, X_train, y_train, eval_set):
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = self.create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders

    def create_dataloaders(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
        num_workers,
        drop_last,
        pin_memory,
    ):
        """
        Create dataloaders with or without subsampling depending on weights and balanced.

        Parameters
        ----------
        X_train : np.ndarray
            Training data
        y_train : np.array
            Mapped Training targets
        eval_set : list of tuple
            List of eval tuple set (X, y)
        weights : either 0, 1, dict or iterable
            if 0 (default) : no weights will be applied
            if 1 : classification only, will balanced class with inverse frequency
            if dict : keys are corresponding class values are sample weights
            if iterable : list or np array must be of length equal to nb elements
                        in the training set
        num_workers : int
            how many subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process
        drop_last : bool
            set to True to drop the last incomplete batch, if the dataset size is not
            divisible by the batch size. If False and the size of dataset is not
            divisible by the batch size, then the last batch will be smaller
        pin_memory : bool
            Whether to pin GPU memory during training

        Returns
        -------
        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
            Training and validation dataloaders
        """
        need_shuffle, sampler = create_sampler(weights, y_train)

        if scipy.sparse.issparse(X_train):
            raise TypeError("Sparse matrix not supported yet")
        else:
            train_dataloader = StockDataLoaderCS(
                StockDatasetCS(X_train.astype(np.float32), y_train),
                sampler=sampler,
                shuffle=need_shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
            )

        valid_dataloaders = []
        for X, y in eval_set:
            if scipy.sparse.issparse(X):
                raise TypeError("Sparse matrix not supported yet")
            else:
                valid_dataloaders.append(
                    StockDataLoaderCS(
                        StockDatasetCS(X.astype(np.float32), y),
                        sampler=sampler,
                        shuffle=need_shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        pin_memory=pin_memory,
                    )
                )

        return train_dataloader, valid_dataloaders

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
        from_unsupervised=None,
        warm_start=False,
        augmentations=None,
        compute_importance=True,
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
            dict for custom weights per class
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        from_unsupervised: unsupervised trained model
            Use a previously self supervised model as starting weights
        warm_start: bool
            If True, current model parameters are used to start training
        compute_importance : bool
            Whether to compute feature importance
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")
        self.augmentations = augmentations
        self.compute_importance = compute_importance

        if self.augmentations is not None:
            # This ensure reproducibility
            self.augmentations._set_seed()

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        # check_input(X_train)
        check_warm_start(warm_start, from_unsupervised)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set_stocks(
            eval_set, eval_name, X_train, y_train
        )

        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )

        if from_unsupervised is not None:
            # Update parameters to match self pretraining
            self.__update__(**from_unsupervised.get_params())

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        if from_unsupervised is not None:
            self.load_weights_from_unsupervised(from_unsupervised)
            warnings.warn("Loading weights from unsupervised pretraining")
        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        if self.compute_importance:
            # compute feature importance once the best model is defined
            self.feature_importances_ = self._compute_feature_importances(X_train)

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            raise TypeError("Sparse matrix not supported yet")
        else:
            dataloader = StockPredictDataLoaderCS(
                StockPredictDatasetCS(X),
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    def explain(self, X, normalize=False):
        """
        Return local explanation

        Parameters
        ----------
        X : tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data
        normalize : bool (default False)
            Wheter to normalize so that sum of features are equal to 1

        Returns
        -------
        M_explain : matrix
            Importance per sample, per columns.
        masks : matrix
            Sparse matrix showing attention masks used by network.
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            raise TypeError("Sparse matrix not supported yet")
        else:
            dataloader = StockPredictDataLoaderCS(
                StockPredictDatasetCS(X),
                shuffle=False,
            )

        res_explain = []

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )
            original_feat_explain = csc_matrix.dot(
                M_explain.cpu().detach().numpy(), self.reducing_matrix
            )
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, res_masks
