# utils/criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_note_type=1.0, cost_instrument=1.0, cost_pitch=1.0, cost_regression=1.0):
        super().__init__()
        self.cost_note_type = cost_note_type
        self.cost_instrument = cost_instrument
        self.cost_pitch = cost_pitch
        self.cost_regression = cost_regression
        assert cost_note_type != 0 or cost_instrument != 0 or cost_pitch != 0 or cost_regression != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Model outputs with keys 'pred_note_type', 'pred_instrument', 'pred_pitch', 'pred_regression'.
                            Each value is a tensor of shape [batch_size, num_queries, ...].
            targets (dict): Ground truth with keys 'note_type', 'instrument', 'pitch', 'start_time', 'duration', 'velocity'.
                           Each value is a tensor of shape [batch_size, num_targets].

        Returns:
            List of size batch_size, containing tuples of (pred_indices, target_indices).
        """
        bs, num_queries = outputs['pred_note_type'].shape[:2]
        all_indices = []

        for b in range(bs):
            tgt_note = targets['note_type'][b]  # [num_targets]
            tgt_inst = targets['instrument'][b]  # [num_targets]
            tgt_pitch = targets['pitch'][b]  # [num_targets]
            tgt_reg = torch.stack([
                targets['start_time'][b],
                targets['duration'][b],
                targets['velocity'][b]
            ], dim=-1)  # [num_targets, 3]

            num_targets = tgt_note.size(0)
            if num_targets == 0:
                all_indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Compute cost matrices
            # Note Type Cost
            pred_note = outputs['pred_note_type'][b]  # [num_queries, 2]
            cost_note = -pred_note[:, tgt_note]  # [num_queries, num_targets]

            # Instrument Cost
            pred_inst = outputs['pred_instrument'][b]  # [num_queries, 5]
            cost_inst = -pred_inst[:, tgt_inst]  # [num_queries, num_targets]

            # Pitch Cost
            pred_pitch = outputs['pred_pitch'][b]  # [num_queries, 88]
            cost_pitch = -pred_pitch[:, tgt_pitch]  # [num_queries, num_targets]

            # Regression Cost (L1)
            pred_reg = outputs['pred_regression'][b]  # [num_queries, 3]
            # Expand dimensions to [num_queries, num_targets, 3]
            pred_reg_exp = pred_reg.unsqueeze(1).expand(-1, num_targets, -1)
            tgt_reg_exp = tgt_reg.unsqueeze(0).expand(num_queries, -1, -1)
            cost_reg = F.l1_loss(pred_reg_exp, tgt_reg_exp, reduction='none').sum(-1)  # [num_queries, num_targets]

            # Total Cost
            C = self.cost_note_type * cost_note + \
                self.cost_instrument * cost_inst + \
                self.cost_pitch * cost_pitch + \
                self.cost_regression * cost_reg  # [num_queries, num_targets]

            C = C.cpu().numpy()  # Convert to numpy for linear_sum_assignment
            row, col = linear_sum_assignment(C)
            row = torch.as_tensor(row, dtype=torch.int64)
            col = torch.as_tensor(col, dtype=torch.int64)

            # back to original device
            row = row.to(pred_note.device)
            col = col.to(pred_note.device)
            all_indices.append((row, col))

        return all_indices


class CustomCriterion(nn.Module):
    def __init__(self, config=None):
        super(CustomCriterion, self).__init__()
        cost_note_type = config.get('cost_note_type', 1)
        cost_instrument = config.get('cost_instrument', 1)
        cost_pitch = config.get('cost_pitch', 1)
        cost_regression = config.get('cost_regression', 1)
        
        self.matcher = HungarianMatcher(
            cost_note_type=cost_note_type, 
            cost_instrument=cost_instrument, 
            cost_pitch=cost_pitch, 
            cost_regression=cost_regression
        )
        
        self.num_note_types = config['num_classes']['note_type']
        self.num_instruments = config['num_classes']['instrument']
        self.num_pitches = config['num_classes']['pitch']
        
        # Include one extra class for "no object"
        self.note_type_loss = nn.CrossEntropyLoss()
        self.instrument_loss = nn.CrossEntropyLoss()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.L1Loss()
        
        # Loss weights
        self.loss_weights = {
            'note_type': cost_note_type,
            'instrument': cost_instrument,
            'pitch': cost_pitch,
            'regression': cost_regression
        }

    def forward(self, outputs, targets):
        # Perform matching
        indices = self.matcher(outputs, targets)
        
        batch_size, num_queries = outputs['pred_note_type'].shape[:2]
        
        # Initialize target class labels for note_type, instrument, and pitch with "no object" class
        device = outputs['pred_note_type'].device
        num_note_types = self.num_note_types + 1  # Including "no object"
        num_instruments = self.num_instruments + 1  # Including "no object"
        num_pitches = self.num_pitches + 1  # Including "no object"

        # Initialize target classes
        tgt_note_type_classes = torch.full(
            (batch_size, num_queries), 
            fill_value=0,  # Index for "no object"
            dtype=torch.long, 
            device=device
        )

        tgt_instrument_classes = torch.full(
            (batch_size, num_queries), 
            fill_value=0,  # Index for "no object"
            dtype=torch.long, 
            device=device
        )

        tgt_pitch_classes = torch.full(
            (batch_size, num_queries), 
            fill_value=num_pitches-1,  # Index for "no object"
            dtype=torch.long, 
            device=device
        )

        # Prepare regression targets with zeros for unmatched predictions
        tgt_regression = torch.zeros(
            (batch_size, num_queries, 3),
            dtype=outputs['pred_regression'].dtype,
            device=device
        )
        
        # Iterate over each batch sample
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue  # No targets for this sample
            
            # Update target classes for matched predictions
            tgt_note_type_classes[b, pred_idx] = targets['note_type'][b][tgt_idx]
            tgt_instrument_classes[b, pred_idx] = targets['instrument'][b][tgt_idx]
            tgt_pitch_classes[b, pred_idx] = targets['pitch'][b][tgt_idx]
            
            # Update regression targets for matched predictions
            tgt_regression[b, pred_idx] = torch.stack([
                targets['start_time'][b][tgt_idx],
                targets['duration'][b][tgt_idx],
                targets['velocity'][b][tgt_idx]
            ], dim=-1).to(tgt_regression.dtype)
            # print("TGT REGRESSION: ", tgt_regression[b, pred_idx])
        
        # Flatten tensors for loss computation
        pred_note_type = outputs['pred_note_type'].reshape(-1, num_note_types)
        tgt_note_type_classes = tgt_note_type_classes.view(-1)
        
        pred_instrument = outputs['pred_instrument'].reshape(-1, num_instruments)
        tgt_instrument_classes = tgt_instrument_classes.view(-1)
        
        pred_pitch = outputs['pred_pitch'].reshape(-1, num_pitches)
        tgt_pitch_classes = tgt_pitch_classes.view(-1)
        
        # Compute classification losses over all predictions
        loss_note_type = self.note_type_loss(pred_note_type, tgt_note_type_classes)
        # print("Loss Note Type: ", loss_note_type)
        loss_instrument = self.instrument_loss(pred_instrument, tgt_instrument_classes)
        # print("Loss Instrument: ", loss_instrument)
        # print("default: ", num_pitches)
        loss_pitch = self.pitch_loss(pred_pitch, tgt_pitch_classes)
        # print("Loss Pitch: ", loss_pitch)
        
        
        losses = {
            'loss_note_type': self.loss_weights['note_type'] * loss_note_type,
            'loss_instrument': self.loss_weights['instrument'] * loss_instrument,
            'loss_pitch': self.loss_weights['pitch'] * loss_pitch
        }

        # Compute regression loss only for matched predictions
        matched_preds = []
        matched_tgts = []
        
        for b, (pred_idx, _) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_preds.append(outputs['pred_regression'][b][pred_idx])
            matched_tgts.append(tgt_regression[b][pred_idx])
        
        if matched_preds:
            pred_regression_matched = torch.cat(matched_preds, dim=0)
            tgt_regression_matched = torch.cat(matched_tgts, dim=0)
            # print("Start Regression Loss")
            loss_regression = self.regression_loss(pred_regression_matched, tgt_regression_matched)
            losses['loss_regression'] = self.loss_weights['regression'] * loss_regression
        else:
            losses['loss_regression'] = torch.tensor(0.0, device=device)

        # Initialize debuginfo dictionary
        debuginfo = {}

        with torch.no_grad():
            epsilon = 1e-7  # Small value to avoid division by zero

            # Function to compute accuracy and min F1 for a classification head
            def compute_metrics(pred_logits, tgt_classes, num_classes, no_object_idx):
                pred_labels = pred_logits.argmax(dim=1)  # [N]
                valid = tgt_classes != no_object_idx  # [N]
                correct = (pred_labels[valid] == tgt_classes[valid]).sum().float()
                total = valid.sum().float()
                accuracy = correct / (total + epsilon)

                f1_scores = []
                for cls in range(1, num_classes):  # Exclude "no object" class
                    true_positive = ((pred_labels == cls) & (tgt_classes == cls)).sum().float()
                    predicted_positive = (pred_labels == cls).sum().float()
                    actual_positive = (tgt_classes == cls).sum().float()

                    precision = true_positive / (predicted_positive + epsilon)
                    recall = true_positive / (actual_positive + epsilon)
                    f1 = 2 * precision * recall / (precision + recall + epsilon)
                    f1_scores.append(f1)

                if f1_scores:
                    min_f1 = min(f1_scores)
                else:
                    min_f1 = torch.tensor(0.0, device=device)

                return accuracy, min_f1

            # Compute metrics for note_type
            note_acc, NoteMinF1 = compute_metrics(
                pred_logits=pred_note_type,
                tgt_classes=tgt_note_type_classes,
                num_classes=num_note_types,
                no_object_idx=0
            )
            debuginfo['note_acc'] = note_acc
            debuginfo['NoteMinF1'] = NoteMinF1

            # Compute metrics for instrument
            InstAcc, InstMinF1 = compute_metrics(
                pred_logits=pred_instrument,
                tgt_classes=tgt_instrument_classes,
                num_classes=num_instruments,
                no_object_idx=0
            )
            debuginfo['InstAcc'] = InstAcc
            debuginfo['InstMinF1'] = InstMinF1

            # Compute metrics for pitch
            pit_acc, pit_MF1 = compute_metrics(
                pred_logits=pred_pitch,
                tgt_classes=tgt_pitch_classes,
                num_classes=num_pitches,
                no_object_idx=num_pitches - 1
            )
            debuginfo['pit_acc'] = pit_acc
            debuginfo['pit_MF1'] = pit_MF1

            # Compute MSE for each regression task
            if matched_preds:
                mse_start_time = F.mse_loss(pred_regression_matched[:, 0], tgt_regression_matched[:, 0], reduction='mean')
                mse_duration = F.mse_loss(pred_regression_matched[:, 1], tgt_regression_matched[:, 1], reduction='mean')
                mse_velocity = F.mse_loss(pred_regression_matched[:, 2], tgt_regression_matched[:, 2], reduction='mean')
            else:
                mse_start_time = torch.tensor(0.0, device=device)
                mse_duration = torch.tensor(0.0, device=device)
                mse_velocity = torch.tensor(0.0, device=device)

            debuginfo['R_M_ST'] = mse_start_time
            debuginfo['R_M_dur'] = mse_duration
            debuginfo['R_M_v'] = mse_velocity

        return losses, debuginfo
