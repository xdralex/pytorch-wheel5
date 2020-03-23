from typing import List

from .loop import *
from ..tracking import TrialTracker, FitState


def fit(device: Union[torch.device, int],
        model: Module,
        classes: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        ctrl_loader: DataLoader,
        train_loss: Module,
        eval_loss: Module,
        train_accuracy: 'Accuracy',
        eval_accuracy: 'Accuracy',
        optimizer: Optimizer,
        scheduler: Optional[Any],
        group_names: List[str],
        num_epochs: int,
        tracker: Optional[TrialTracker] = None,
        display_progress: bool = True,
        sampled_epochs=0,
        samples=8):
    dummy_train_handler = TrainEvalEpochHandler('dummy-train', 1, accuracy=train_accuracy, sampled_epochs=sampled_epochs + 1, samples=samples)
    dummy_val_handler = TrainEvalEpochHandler('dummy-val', 1, accuracy=eval_accuracy, sampled_epochs=sampled_epochs + 1, samples=samples)
    dummy_ctrl_handler = TrainEvalEpochHandler('dummy-ctrl', 1, accuracy=eval_accuracy, sampled_epochs=sampled_epochs + 1, samples=samples)

    main_train_handler = TrainEvalEpochHandler('train', num_epochs, accuracy=train_accuracy, sampled_epochs=sampled_epochs, samples=samples)
    main_val_handler = TrainEvalEpochHandler('val', num_epochs, accuracy=eval_accuracy, sampled_epochs=sampled_epochs, samples=samples)
    main_ctrl_handler = TrainEvalEpochHandler('ctrl', num_epochs, accuracy=eval_accuracy, sampled_epochs=sampled_epochs, samples=samples)

    for epoch in range(0, num_epochs + 1):
        if epoch == 0:
            train_handler, val_handler, ctrl_handler = dummy_train_handler, dummy_val_handler, dummy_ctrl_handler
            train_optimizer, train_scheduler = None, None
        else:
            train_handler, val_handler, ctrl_handler = main_train_handler, main_val_handler, main_ctrl_handler
            train_optimizer, train_scheduler = optimizer, scheduler

        train_metrics = run_epoch(device, model, train_loader, train_loss, train_optimizer, train_scheduler, train_handler, display_progress=display_progress)
        val_metrics = run_epoch(device, model, val_loader, eval_loss, None, None, val_handler, display_progress=display_progress)
        ctrl_metrics = run_epoch(device, model, ctrl_loader, eval_loss, None, None, ctrl_handler, display_progress=display_progress)

        if tracker.tensorboard_cfg.track_predictions:
            predict_handler = PredictEpochHandler()
            prediction = run_epoch(device, model, val_loader, None, None, None, predict_handler, display_progress=display_progress)
        else:
            prediction = None

        if tracker:
            tracker.epoch_completed(FitState(model=model,
                                             train_loss=train_loss,
                                             eval_loss=eval_loss,
                                             optimizer=optimizer,
                                             epoch=epoch,
                                             num_epochs=num_epochs,
                                             train_metrics=train_metrics,
                                             val_metrics=val_metrics,
                                             ctrl_metrics=ctrl_metrics),
                                    train_samples=train_handler.random_samples_meter.value(),
                                    val_samples=val_handler.random_samples_meter.value(),
                                    ctrl_samples=ctrl_handler.fixed_samples_meter.value(),
                                    classes=classes,
                                    prediction=prediction,
                                    prediction_dataset=val_loader.dataset,
                                    optimizer_group_names=group_names)


def score_blend(device: Union[torch.device, int],
                models: List[Module],
                loader: DataLoader,
                eval_loss: Module,
                display_progress: bool = True) -> Dict[str, Union[int, float]]:
    assert len(models) > 0

    y = None
    y_probs_list = []

    for model in models:
        model_device = model.to(device)

        handler = PredictEpochHandler()
        results = run_epoch(device, model_device, loader, None, None, None, handler, display_progress=display_progress)

        order = torch.argsort(results.indices)
        y_ordered = torch.index_select(results.y, dim=0, index=order)
        y_probs_ordered = torch.index_select(results.y_probs, dim=0, index=order)

        if y is None:
            y = y_ordered
        else:
            assert bool(torch.eq(y, y_ordered).all())

        y_probs_list.append(y_probs_ordered)

        del model

    y_probs_stack = torch.stack(y_probs_list, dim=0)
    y_probs_blend = torch.mean(y_probs_stack, dim=0)

    y_hat = torch.argmax(y_probs_blend, dim=1)
    loss_value = float(eval_loss(y_probs_blend, y))

    correct = float(torch.sum(y_hat == y))
    total = float(y.shape[0])

    acc = correct / total

    return {'loss': loss_value, 'acc': acc}
