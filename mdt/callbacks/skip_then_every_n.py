from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint


class SkipThenEveryNEpochsCheckpoint(ModelCheckpoint):
	"""
	Save checkpoints every `every_n_epochs` epochs, but only after skipping the
	first `skip_first_n_epochs` epochs.

	Example:
		- skip_first_n_epochs=10, every_n_epochs=5 -> save at epochs 15, 20, 25, ...

	Notes:
		- We set base ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=True)
		  and gate the saving condition here, so super() will perform the actual save
		  only on the epochs we allow.
		- Epoch numbers in messages are 1-based (trainer.current_epoch + 1).
	"""

	def __init__(
		self,
		skip_first_n_epochs: int = 0,
		every_n_epochs: int = 5,
		**kwargs,
	) -> None:
		# ensure base checkpoint checks pass on epochs we forward to it
		kwargs = dict(kwargs)
		kwargs.setdefault("every_n_epochs", 1)
		kwargs.setdefault("save_on_train_epoch_end", True)
		super().__init__(**kwargs)
		if every_n_epochs <= 0:
			raise ValueError("every_n_epochs must be > 0")
		if skip_first_n_epochs < 0:
			raise ValueError("skip_first_n_epochs must be >= 0")
		self.skip_first_n_epochs = int(skip_first_n_epochs)
		self.save_every_n_after_skip = int(every_n_epochs)

	def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
		epoch_idx = int(trainer.current_epoch) + 1  # Lightning uses 0-based internally
		# Skip first N epochs entirely
		if epoch_idx <= self.skip_first_n_epochs:
			return
		# Save only on the desired cadence after the skip window
		if (epoch_idx - self.skip_first_n_epochs) % self.save_every_n_after_skip == 0:
			return super().on_train_epoch_end(trainer, pl_module)
		# otherwise do nothing this epoch
		return

