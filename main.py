import argparse
from copy import deepcopy

import torch

import misc
import optimizers
from networks import MLP, CNN, fit


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=10)
	parser.add_argument('-dataset', type=str, default='cifar')
	parser.add_argument('-num_train', type=int, default=50000)
	parser.add_argument('-num_val', type=int, default=2048)
	parser.add_argument('-lr_schedule', action='store_true', default=True)
	parser.add_argument('-only_plot', type=str, default='False')  # Changed default to False to run training
	parser.add_argument('-batch_size', type=int, default=256, help='Training batch size')
	parser.add_argument('-num_workers', type=int, default=4, help='Number of data loading workers')
	parser.add_argument('-no_mixed_precision', action='store_true', help='Disable mixed precision training')
	parser.add_argument('-gradient_accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation')
	parser.add_argument('-no_monitor_memory', action='store_true', help='Disable GPU memory monitoring')
	args = parser.parse_args()

	# Convert string to boolean values
	if isinstance(args.only_plot, str):
		args.only_plot = args.only_plot.lower() != 'false'
	if isinstance(args.lr_schedule, str) and args.lr_schedule.lower() == 'false':
		args.lr_schedule = False
	
	# Check if CUDA is available
	if torch.cuda.is_available():
		print(f"CUDA available: {torch.cuda.is_available()}")
		print(f"CUDA device count: {torch.cuda.device_count()}")
		print(f"Current CUDA device: {torch.cuda.current_device()}")
		print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
		print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
	else:
		print("CUDA not available, using CPU.")
		args.no_mixed_precision = True

	data = getattr(misc, 'load_'+args.dataset)(
		num_train=args.num_train,
		num_val=args.num_val
	)

	print(f'Loaded data partitions: ({len(data[0])}), ({len(data[1])})')
	
	# Print whether we're training or just plotting
	print(f"Mode: {'Plotting only' if args.only_plot else 'Training and plotting'}")
	
	opt_tasks = [
		'sgd',
		'sgd_momentum',
		'sgd_nesterov',
		'sgd_weight_decay',
		'sgd_lrd',
		'rmsprop',
		'adam',
		'adam_l2',
		'adamW',
		'adam_lrd',
		'Radam',
		'RadamW',
		'Radam_lrd',
		'nadam',
		'lookahead_sgd',
		'lookahead_adam',
		'gradnoise_adam',
		'graddropout_adam',
		'talt',           # Add TALT optimizer
		'talt_lrd'        # Add TALT with Learning Rate Dropout
	]
	opt_losses, opt_val_losses, opt_labels = [], [], []

	def do_stuff(opt):
		print(f'\nTraining {opt} for {args.num_epochs} epochs...')
		net = CNN() if args.dataset == 'cifar' else MLP()
		_, kwargs = misc.split_optim_dict(misc.optim_dict[opt])
		optimizer = misc.task_to_optimizer(opt)(
			params=net.parameters(),
			**kwargs
		)
		optimizer = misc.wrap_optimizer(opt, optimizer)

		return fit(
			net, 
			data, 
			optimizer, 
			batch_size=args.batch_size,
			num_epochs=args.num_epochs, 
			lr_schedule=args.lr_schedule,
			num_workers=args.num_workers,
			pin_memory=torch.cuda.is_available(),
			mixed_precision=not args.no_mixed_precision,
			gradient_accumulation_steps=args.gradient_accumulation_steps,
			monitor_memory=not args.no_monitor_memory
		)

	for opt in opt_tasks:
		try:
			print(f"Processing optimizer: {opt}")
			if args.only_plot:
				losses = misc.load_losses(dataset=args.dataset, filename=opt)
				val_losses = misc.load_losses(dataset=args.dataset, filename=opt+'_val')
			else:
				print(f"Starting training for {opt}...")
				losses, val_losses = do_stuff(opt)
				print(f"Training completed for {opt}, saving results...")
				misc.save_losses(losses, dataset=args.dataset, filename=opt)
				misc.save_losses(val_losses, dataset=args.dataset, filename=opt+'_val')
				print(f"Results saved for {opt}")

			if losses is not None:
				opt_losses.append(losses)
				opt_val_losses.append(val_losses)
				opt_labels.append(misc.split_optim_dict(misc.optim_dict[opt])[0])
		except Exception as e:
			print(f"Error processing {opt}: {str(e)}")
			import traceback
			traceback.print_exc()

	if not torch.cuda.is_available():
		assert len(opt_losses) == len(opt_val_losses)
		misc.plot_losses(
			losses=opt_losses,
			val_losses=opt_val_losses,
			labels=opt_labels,
			num_epochs=args.num_epochs,
			title=args.dataset,
			plot_val=False,
			yscale_log=False,
			max_epochs=30
		)
