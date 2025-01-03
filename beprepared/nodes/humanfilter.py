import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
from .utils import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import numpy as np

from beprepared.image import Image
from beprepared.workspace import Workspace, Abort
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty
from beprepared.web import WebInterface
from beprepared.web import Applet
from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse

class HumanFilterApplet(Applet):
    '''Backend for the HumanFilter and SmartHumanFilter nodes. This class is not intended to be used directly.'''
    def __init__(self, images_to_filter, domain='default', cb_filtered=lambda image: None):
        super().__init__('humanfilter', 'HumanFilter')
        self.images_to_filter = images_to_filter
        self.domain = domain
        self.cb_filtered = cb_filtered

        # Map image IDs to images and properties
        @self.app.get("/api/images")
        def get_images():
            images_data = [{"id": idx, "objectid": image.objectid.value } 
                            for idx,image in enumerate(self.images_to_filter)
                            if not image.passed_human_filter.has_value]
            random.shuffle(images_data)
            return {"images": images_data, "domain": self.domain}

        @self.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = Workspace.current.get_object_path(object_id)
            return FileResponse(path)

        @self.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            action = data.get('action')
            if action not in ['accept', 'reject']:
                return JSONResponse({"error": "Invalid action"}, status_code=400)
            image = self.images_to_filter[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            if action == 'reject':
                image.passed_human_filter.value = False
            elif action == 'accept':
                image.passed_human_filter.value = True
            else:
                return JSONResponse({"error": "Invalid action"}, status_code=400)
            if self.cb_filtered(image):
                return {"status": "done"}
            else:
                return {"status": "ok"}

class HumanFilter(Node):
    '''HumanFilter presents a web-based UI to enable a human to manually filter images for inclusion in a dataset.

    This is most commonly used when scraped or automatically collected images are to be used, as quality is often variable and high quality
    data is essential for good training results.

    This UI enables a human to filter ~5k images in an hour or two, which is quite practical for an individual. If you have much larger data sets
    consider using `SmartHumanFilter` instead. 
    '''

    def __init__(self, domain: str = 'default', skip_ui: bool = False):
        ''' Initialize a HumanFilter node

        Args:
        - domain (str): Domain to use for caching the results. This interoperates with `SmartHumanFilter` domains. Most people should leave this as 'default' but if your workflow contains multiple HumanFilter steps in your workflow that use different filter criteria and may encounter the same images, you should assign a unique domain to each.
        - skip_ui (bool): If True, applies previously established filters but passes all unfiltered images to the next stage without showing the UI. Useful for testing or when you want to temporarily bypass human filtering.
        '''
        super().__init__()
        self.domain = domain
        self.skip_ui = skip_ui

    def eval(self, dataset):
        images_to_filter = []
        for image in dataset.images:
            image.passed_human_filter = CachedProperty('humanfilter', image, domain=self.domain)
            if not image.passed_human_filter.has_value:
                images_to_filter.append(image)

        if len(images_to_filter) == 0:
            self.log.info("All images already have been filtered, skipping")
            dataset.images = [image for image in dataset.images if image.passed_human_filter.value]
            return dataset

        if self.skip_ui:
            self.log.info(f"Skipping UI for {len(images_to_filter)} unfiltered images")
            dataset.images = [image for image in dataset.images if image.passed_human_filter.value or not image.passed_human_filter.has_value]
            return dataset

        def desc():
            accepted_count = len([image for image in dataset.images if image.passed_human_filter.has_value and image.passed_human_filter.value])
            filtered_count = len([image for image in dataset.images if image.passed_human_filter.has_value])
            if filtered_count > 0:
                return f"Human filter ({accepted_count/filtered_count*100:.1f}% accepted)"
            else:
                return "Human filter"

        self.log.info(f"Filtering images using human filter for {len(images_to_filter)} images (already filtered: {len(dataset.images) - len(images_to_filter)})")   

        # Run webui with progress bar 
        progress_bar = tqdm(total=len(dataset.images), desc=desc())
        progress_bar.n = len([image for image in dataset.images if image.passed_human_filter.has_value])
        progress_bar.refresh()
        def image_filtered(image: Image):
            progress_bar.n += 1
            progress_bar.set_description(desc())
            progress_bar.refresh()
        applet = HumanFilterApplet(images_to_filter, domain=self.domain, cb_filtered=image_filtered)
        applet.run(self.workspace)
        progress_bar.close()

        total_count    = len(dataset.images)
        accepted_count = len([image for image in dataset.images if image.passed_human_filter.value])
        self.log.info(f"Human filtering completed, accepted {accepted_count} out of {total_count} images ({accepted_count/total_count*100:.1f}%)")

        dataset.images = [image for image in dataset.images if image.passed_human_filter.value]

        return dataset

class ImageDataset(TorchDataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[2], 2)
        )
    
    def forward(self, x):
        return self.network(x)


class SmartHumanFilter(Node):
    '''
    **Experimental**

    SmartHumanFilter uses machine learning to filter images. This is most useful for large datasets, at least 5,000 images, as you need a fairly large number 
    of images per class (accepted/rejected) to train a high quality model. If you do not have enough images, just use HumanFilter instead. You'll get 100% 
    accuracy with less time investment.

    First, it presents the web-based "Human filter" UI until it has collected at minimum number images for each of the 'rejected' and 'accepted' classes.

    Once it has collected those images, it trains a classifier model, that uses CLIP embeddings to predict image acceptance/rejection. 

    Afterwards, it uses that model to predict results for the remaining images. If the model is not confident enough in its prediction, it will either 
    reject the image, accept the image, or the web-based UI will be presented to a human to make the final decision.

    This model is marked as **Experimental**. In test, we have achieved ~93% accuracy with this model, but since hand-labeling thousands of examples is time-consuming, we have not performed many tests.
    We welcome feedback on how well this approach works for you.

    With large data sets (e.g. 50k+ images), this model can save a massive amount of human effort at minimal cost to dataset quality.
    '''

    def __init__(self, 
                 domain: str = 'default', 
                 model_version: str = 'v1',
                 min_images: int = 5000,
                 min_per_class: int = 1000, 
                 learning_rate: float = 1e-3, 
                 batch_size: int = 128, 
                 epochs: int = 500, 
                 patience: int = 20, 
                 dropout: float = 0.5,
                 min_confidence: float = 0.7,
                 min_accuracy: float = 0.9,
                 when_uncertain: Literal['reject', ' accept', 'human'] = 'reject'):
        ''' Initialize a SmartHumanFilter node
        Args:

        - domain (str)= 'default' - Domain to use for caching the results. This interoperates with `HumanFilter` domains.
        - model_version (str): str = 'v1' - Version for the model. Change this if you want to force the model to be re-trained.
        - min_images (int) = 5000 - Minimum number of images required to make SmartHumanFilter worthwhile. If you have less images, use HumanFilter instead.
        - min_per_class (int) = 1000 - Minimum number of images required for each class (accepted/rejected) before training the model.
        - learning_rate (float) = 1e-3 - Learning rate for the model training
        - batch_size (int) = 128 - Batch size for the model training
        - epochs (int) = 500 - Number of epochs for the model training
        - patience (int) = 20 - Patience for early stopping. If the validation accuracy does not improve for this many epochs, the training stops.
        - dropout (float) = 0.5 - Dropout rate for the model
        - min_accuracy (float) = 0.9 - Minimum accuracy required for the model to be considered good enough.
        - min_confidence (float) = 0.7 - Minimum confidence required for the model to accept the prediction. If the model is not confident enough, `when_uncertain` determines how the image is handled.
        - when_uncertain ('reject' | ' accept' | 'human') = 'reject' - What to do when the model's predictions do not exceed `min_confidence`. 'reject' will reject the image and 'accept' will accept the image. If this is set to `human`, then the web interface will be presented to filter the uncertain images manually. If you have a large dataset, 'reject' is almost certainly the best choice, as including bad images is much more harmful than excluding good ones.
        as '''
        super().__init__()
        self.domain = domain
        self.model_version = model_version
        self.min_per_class = min_per_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout = dropout
        self.min_accuracy = min_accuracy
        self.min_confidence = min_confidence
        self.when_uncertain = when_uncertain
        self.min_images = min_images
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_training_data(self, dataset: Dataset):
        for image in dataset.images:
            if not image.clip.has_value:
                raise Abort("SmartHumanFilter requires images to have clip embeddings. Run ClipEmbedding first.")

        accepted_count = len([image for image in dataset.images if image.passed_human_filter.has_value and image.passed_human_filter.value])
        rejected_count = len([image for image in dataset.images if image.passed_human_filter.has_value and not image.passed_human_filter.value])

        # Run webui and collect more data if needed
        if accepted_count < self.min_per_class or rejected_count < self.min_per_class:
            def cb_filtered(image: Image):
                accepted_count = len([image for image in dataset.images if image.passed_human_filter.has_value and image.passed_human_filter.value])
                rejected_count = len([image for image in dataset.images if image.passed_human_filter.has_value and not image.passed_human_filter.value])
                return accepted_count >= self.min_per_class and rejected_count >= self.min_per_class

            applet = HumanFilterApplet([image for image in dataset.images if not image.passed_human_filter.has_value], domain=self.domain, cb_filtered=cb_filtered)
            applet.run(self.workspace)

        # Prepare dataset
        embeddings = []
        labels = []
        for img in dataset.images:
            if img.passed_human_filter.has_value:
                labels.append(int(img.passed_human_filter.value))
                embeddings.append(img.clip.value)

        self.embedding_size = len(embeddings[0])
        self.log.info("Prepared data set")
        self.log.info(f"Accepted: {sum(labels)} Rejected: {len(labels) - sum(labels)}")
        self.log.info(f"Embedding size: {self.embedding_size}")

        return self.upsample_training_data(np.array(embeddings), np.array(labels))

    def upsample_training_data(self, embeddings, labels):
        # Separate classes
        accepted = embeddings[labels == 1]
        rejected = embeddings[labels == 0]
        accepted_labels = labels[labels == 1]
        rejected_labels = labels[labels == 0]
        
        # Determine minority and majority classes
        if len(accepted) > len(rejected):
            majority_data, majority_labels = accepted, accepted_labels
            minority_data, minority_labels = rejected, rejected_labels
        else:
            majority_data, majority_labels = rejected, rejected_labels
            minority_data, minority_labels = accepted, accepted_labels
        
        # Upsample minority class
        minority_data_upsampled, minority_labels_upsampled = resample(
            minority_data,
            minority_labels,
            replace=True,
            n_samples=len(majority_data)
        )
        
        # Combine majority and upsampled minority class
        upsampled_embeddings = np.vstack((majority_data, minority_data_upsampled))
        upsampled_labels = np.hstack((majority_labels, minority_labels_upsampled))
        
        # Shuffle the upsampled dataset
        indices = np.arange(len(upsampled_labels))
        np.random.shuffle(indices)
        upsampled_embeddings = upsampled_embeddings[indices]
        upsampled_labels = upsampled_labels[indices]

        self.log.info(f"resampled data, expanded minority class from {len(minority_data)} to {len(minority_data_upsampled)}")
        
        return upsampled_embeddings, upsampled_labels
    
    def train_network(self, training_data, validation_data):
        train_dataset = ImageDataset(*training_data)
        val_dataset = ImageDataset(*validation_data)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        model = SimpleMLP(input_size=self.embedding_size, dropout=self.dropout).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_val_acc = 0.0
        patience_counter = 0

        val_loss, val_acc, _, _ = self.eval_model(model, val_loader)
        self.log.info(f"Baseline: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        model_path = os.path.join(Workspace.current.tmp_dir, f"smartfilter_model_{self.domain}.pth")
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            
            val_loss, val_acc, all_labels, all_preds = self.eval_model(model, val_loader)
            self.log.info(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.log.info("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model, model_path

    def eval_model(self, model, loader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []
        threshold_preds = []  # New preds including "unknown"
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)  # Original predicted classes
                unknown_preds = np.where(np.max(probs, axis=1) > self.min_confidence, preds, 2)  # 2 for "unknown"
                
                all_preds.extend(preds)  # Original preds used for loss/accuracy
                threshold_preds.extend(unknown_preds)  # Preds with "unknown"
                all_labels.extend(targets.cpu().numpy())
        
        avg_loss = running_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)  # Use original predictions for accuracy

        return avg_loss, accuracy, all_labels, threshold_preds  # Return min_confidence-modified preds

    def print_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds, labels=[1, 0, 2])  # 1: Accepted, 0: Rejected, 2: Unknown
        tp, fn, fp, tn, _, _ = cm.ravel() if cm.size == 4 else (cm[1,1], cm[1,0], cm[0,1], cm[0,0], cm[2,1:].sum(), cm[:2,2].sum())
        
        matrix = f"""
Confusion Matrix:
               Predicted
               Accepted  Rejected  Unknown
Actual Accepted    {tp:5}      {fn:5}      {cm[1,2]:5}
       Rejected    {fp:5}      {tn:5}      {cm[0,2]:5}
       Unknown     {cm[2,1]:5}    {cm[2,0]:5}      {cm[2,2]:5}
"""
        self.log.info(matrix)

    
    def eval(self, dataset):
        if len(dataset.images) < self.min_images:
            raise Abort(f"SmartHumanFilter is not useful for small data sets, use HumanFilter instead. Dataset size: {len(dataset.images)}")

        if not all(image.clip.has_value for image in dataset.images):
            raise Abort("SmartHumanFilter requires images to have clip embeddings. Add a ClipEmbedding step.")

        for image in dataset.images:
            image.passed_human_filter = CachedProperty('humanfilter', image, domain=self.domain)
            image.passed_smart_filter = CachedProperty('smartfilter', image, domain=self.domain)

        embedding_size = len(dataset.images[0].clip.value)
        hyperparameters = {
            'embedding_size': embedding_size,
            'min_per_class': self.min_per_class,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'dropout': self.dropout,
        }
        prop_model = CachedProperty('smartfilter', self.model_version, hyperparameters, domain=self.domain)
        if prop_model.has_value:
            model_file = Workspace.current.get_object_path(prop_model.value)
            model = SimpleMLP(input_size=embedding_size, dropout=self.dropout)
            model.load_state_dict(torch.load(model_file, weights_only=True))
            model.to(self.device)
        else:
            embeddings, labels = self.get_training_data(dataset)
            dataset_size = len(labels)
            split = int(0.8 * dataset_size)
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            train_indices, val_indices = indices[:split], indices[split:]
            
            training_data = (embeddings[train_indices], labels[train_indices])
            validation_data = (embeddings[val_indices], labels[val_indices])
            
            model, model_path = self.train_network(training_data, validation_data)
        
            test_embeddings, test_labels = validation_data
            test_dataset = ImageDataset(test_embeddings, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            test_loss, test_acc, test_labels, test_preds = self.eval_model(model, test_loader)
            if test_acc < self.min_accuracy:
                raise Abort(f"Accuracy is low ({test_acc:.4f}), improve your data set or hyperparameters and try again")

            self.log.info("-"*80)
            self.log.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            self.print_confusion_matrix(test_labels, test_preds)
            self.log.info("-"*80)

            prop_model.value = self.workspace.put_object(model_path)

        # use model to predict filter results for unknown images using image.clip.value embedding
        unknown_images = [image for image in dataset.images if not image.passed_human_filter.has_value]
        unknown_embeddings = np.array([image.clip.value for image in unknown_images])
        unknown_dataset = ImageDataset(unknown_embeddings, np.zeros(len(unknown_embeddings)))
        unknown_loader = DataLoader(unknown_dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        idx = 0  
        with torch.no_grad():
            for inputs, _ in unknown_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                batch_size = len(preds)  
                for i in range(batch_size):
                    image = unknown_images[idx]
                    if np.max(probs[i]) > self.min_confidence:  
                        image.passed_smart_filter.value = bool(preds[i])
                    idx += 1  

        remaining_images = [image for image in unknown_images if not image.passed_smart_filter.has_value]
        self.log.info(f"after predictions, {len(remaining_images)} images remain uncertain")

        if self.when_uncertain == 'accept':
            dataset.images = [image for image in dataset.images if 
                                (image.passed_human_filter.has_value and image.passed_human_filter.value) or 
                                (image.passed_smart_filter.has_value and image.passed_smart_filter.value) or
                                (not image.passed_smart_filter.has_value and not image.passed_human_filter.has_value)]
        elif self.when_uncertain == 'reject':
            dataset.images = [image for image in dataset.images if 
                                (image.passed_human_filter.has_value and image.passed_human_filter.value) or 
                                (image.passed_smart_filter.has_value and image.passed_smart_filter.value)]
        elif self.when_uncertain == 'human':
            # Run webui with progress bar so the human can fill in the gaps
            # It's probbaly safer in most cases to just reject the remaining images, 
            # so this is disabled for now
            progress_bar = tqdm(total=len(remaining_images), desc="Human Filtering")
            progress_bar.refresh()
            def image_filtered(image: Image):
                progress_bar.n += 1
                progress_bar.refresh()
            applet = HumanFilterApplet(remaining_images, domain=self.domain, cb_filtered=image_filtered)
            applet.run(self.workspace)
            progress_bar.close()

            dataset.images = [image for image in dataset.images if 
                                (image.passed_human_filter.has_value and image.passed_human_filter.value) or 
                                (image.passed_smart_filter.has_value and image.passed_smart_filter.value)]
        else:
            raise Abort(f"Invalid value for when_uncertain in SmartHumanFilter: {self.when_uncertain}")

        return dataset

__all__ = ['HumanFilter', 'SmartHumanFilter']
